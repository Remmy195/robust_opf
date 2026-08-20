"""The frequency screen of Section 4.2: RoCoF, then the nadir.

Two tests, both driven by the net imbalance ``delta_P`` -- surviving demand less
surviving generation, signed so that a deficit is positive.

    eq (10)   RoCoF = f0 |delta_P| / (2 H_sys) <= RoCoF-bar
    nadir     integrate the centre-of-inertia response and reject the case when
              the nadir falls below f_under, the first stage of under-frequency
              load shedding

THE SCREEN RUNS BEFORE MODEL (D).  Section 4.2 orders it that way, and the order
has a consequence worth stating because it constrains what may be built here:
gamma, the response window scale of eq (6d), belongs to (D) and cannot reach the
nadir.  There is therefore no gamma in this module and no gamma sweep in the
frequency path.  A screen that responded to gamma would be reporting an effect
that the method does not contain.

THE DAMPING BASE IS PART OF THE VALUE, NOT A CONVENTION.
--------------------------------------------------------
The swing equation

    2 H_sys d(df)/dt = sum_i dPm_i - delta_P - D df

is on the SYSTEM base: delta_P and H_sys are both p.u. on baseMVA, so D must be
too.  The textbook figure -- load damping of 1 to 2 % of load per 1 % of
frequency -- is on the LOAD base, and the two differ by the load-to-baseMVA
ratio, which is 671 on ACTIVSg2000.  Passing the textbook number raw therefore
under-damps by a factor of 671, and the failure is quiet: the nadir simply comes
out deeper, and the run still finishes.

So the damping is not a float here.  `LoadDamping` carries its base with it, and
`NadirConfig` will not accept a bare number.  A caller must say which quantity
it holds -- `LoadDamping.per_load(1.5)` or `LoadDamping.on_system_base(0.02)` --
and the conversion happens in one place that a test can reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..network import Network
from .dynamics import UnitDynamics

#: Bases the damping can be quoted on.
LOAD_BASE = "load"
SYSTEM_BASE = "system"


class FrequencyError(ValueError):
    """The screen was given something it will not silently interpret."""


###############################################################################
# Load damping
###############################################################################


@dataclass(frozen=True)
class LoadDamping:
    """Load damping D, carrying the base it is quoted on.

    Construct it through one of the two classmethods; the base is never
    defaulted, because the whole point of the type is that a value without its
    base is not a damping.

        LoadDamping.per_load(1.5)         # 1.5 % power per % frequency, the
                                          # textbook figure, on the LOAD base
        LoadDamping.on_system_base(0.02)  # already p.u. on baseMVA

    `on_system` converts, and it is the only thing the integrator ever calls.
    """

    value: float
    base: str

    def __post_init__(self) -> None:
        if self.base not in (LOAD_BASE, SYSTEM_BASE):
            raise FrequencyError(
                f"damping base must be {LOAD_BASE!r} or {SYSTEM_BASE!r}, got "
                f"{self.base!r}")
        if self.value < 0:
            raise FrequencyError(f"load damping cannot be negative, got "
                                 f"{self.value}")

    @classmethod
    def per_load(cls, value: float) -> "LoadDamping":
        """D on the LOAD base: p.u. power per p.u. frequency, of load.

        This is where the textbook 1-2 %/% goes.  It is converted to the system
        base by multiplying by the total demand in p.u., which is the ratio the
        raw value is wrong by.
        """
        return cls(float(value), LOAD_BASE)

    @classmethod
    def on_system_base(cls, value: float) -> "LoadDamping":
        """D already p.u. on baseMVA, the base the swing equation needs."""
        return cls(float(value), SYSTEM_BASE)

    def on_system(self, network: Network) -> float:
        """D on the system base, ready for the swing equation."""
        if self.base == SYSTEM_BASE:
            return self.value
        return self.value * network.load_pu

    def describe(self, network: Network) -> str:
        converted = self.on_system(network)
        if self.base == SYSTEM_BASE:
            return f"D = {converted:g} p.u. on baseMVA (given on the system base)"
        return (f"D = {self.value:g} per unit of load x {network.load_pu:g} p.u. "
                f"load = {converted:g} p.u. on baseMVA")


###############################################################################
# Configuration
###############################################################################


@dataclass(frozen=True)
class ScreenConfig:
    """The screen constants.  None is defaulted from thin air.

    f0, RoCoF-bar and f_under are grid-code quantities and belong in Section 5
    with a citation; the screen refuses to run without them rather than
    supplying a plausible number that would then be reported as if it were data.
    """

    f0_hz: float
    rocof_max_hz_s: float
    f_under_hz: float
    damping: LoadDamping
    horizon_s: float = 30.0
    dt_s: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.damping, LoadDamping):
            raise FrequencyError(
                "damping must be a LoadDamping, not a bare number: the "
                "textbook 1-2 %/% figure is on the LOAD base and the swing "
                "equation needs the SYSTEM base, which differ by the "
                "load-to-baseMVA ratio (671 on ACTIVSg2000). Use "
                "LoadDamping.per_load(...) or LoadDamping.on_system_base(...).")
        for name in ("f0_hz", "rocof_max_hz_s", "f_under_hz", "horizon_s",
                     "dt_s"):
            if getattr(self, name) <= 0:
                raise FrequencyError(f"{name} must be positive, got "
                                     f"{getattr(self, name)}")
        if self.f_under_hz >= self.f0_hz:
            raise FrequencyError(
                f"f_under = {self.f_under_hz} Hz is not below f0 = "
                f"{self.f0_hz} Hz, so every case would be a collapse")
        if self.dt_s > self.horizon_s:
            raise FrequencyError("the time step is longer than the horizon")


@dataclass
class ScreenResult:
    """Whether the surviving system clears both tests, and by how much."""

    cleared: bool
    reason: str
    #: delta_P, p.u. on baseMVA, positive for a generation deficit.
    delta_P_pu: float = 0.0
    #: H_sys, seconds on baseMVA.
    H_sys_s: float = 0.0
    rocof_hz_s: float = 0.0
    nadir_hz: float = float("nan")
    t_nadir_s: float = float("nan")
    n_responsive: int = 0
    failed_test: str = ""          # '', 'rocof' or 'nadir'


###############################################################################
# The two tests
###############################################################################


def system_inertia(network: Network,
                   dynamics: Dict[int, UnitDynamics],
                   live_gens: Iterable[int]) -> float:
    """H_sys in seconds on baseMVA, over the surviving synchronous fleet.

    A unit's H is on its OWN MVA base, so it enters as ``H_g * MBASE_g``.  Where
    the ``.RAW`` MBASE is not available the unit's Pmax stands in for it, which
    is the machine rating to within the unit's own power factor -- stated here
    rather than buried, because it is an approximation and not the data.

    An inverter-interfaced unit carries no H and contributes nothing.  That is
    the physics, not missing data.
    """
    total = 0.0
    for count in live_gens:
        unit = dynamics.get(count)
        if unit is None or not unit.H:
            continue
        mbase = unit.mbase or network.gens[count].Pmax * network.baseMVA
        total += float(unit.H) * float(mbase)
    return total / network.baseMVA


def imbalance(network: Network,
              live_buses: Iterable[int],
              live_gens: Iterable[int],
              P_star: Dict[int, float]) -> float:
    """delta_P: surviving demand less surviving generation, p.u.

    Positive for a deficit, which is the sign convention Section 4.2 states.
    Generation is counted at the PRE-EVENT dispatch, because the disturbance is
    what the loss of those units takes away from a system that was balanced.
    """
    demand = sum(network.buses[bus].Pd for bus in live_buses)
    generation = sum(float(P_star.get(gen, 0.0)) for gen in live_gens)
    return demand - generation


def screen(config: ScreenConfig,
           network: Network,
           dynamics: Dict[int, UnitDynamics],
           live_buses: Iterable[int],
           live_gens: Iterable[int],
           P_star: Dict[int, float]) -> ScreenResult:
    """Both tests, in order.  RoCoF first, since it needs no integration."""
    live_gens = list(live_gens)
    delta_P = imbalance(network, live_buses, live_gens, P_star)
    H_sys = system_inertia(network, dynamics, live_gens)

    if H_sys <= 0.0:
        return ScreenResult(
            cleared=False, failed_test="rocof", delta_P_pu=delta_P, H_sys_s=0.0,
            reason="no surviving synchronous inertia; the centre-of-inertia "
                   "model is undefined and RoCoF is unbounded")

    # ---- eq (10) --------------------------------------------------------
    rocof = config.f0_hz * abs(delta_P) / (2.0 * H_sys)
    if rocof > config.rocof_max_hz_s:
        return ScreenResult(
            cleared=False, failed_test="rocof", delta_P_pu=delta_P,
            H_sys_s=H_sys, rocof_hz_s=rocof,
            reason=f"RoCoF {rocof:.4f} Hz/s exceeds the limit "
                   f"{config.rocof_max_hz_s:.4f} Hz/s")

    # ---- the nadir -------------------------------------------------------
    nadir, t_nadir, n_responsive = integrate_nadir(
        config, network, dynamics, live_gens, P_star, delta_P, H_sys)

    if nadir < config.f_under_hz:
        return ScreenResult(
            cleared=False, failed_test="nadir", delta_P_pu=delta_P,
            H_sys_s=H_sys, rocof_hz_s=rocof, nadir_hz=nadir, t_nadir_s=t_nadir,
            n_responsive=n_responsive,
            reason=f"nadir {nadir:.4f} Hz falls below the first UFLS stage "
                   f"{config.f_under_hz:.4f} Hz")

    return ScreenResult(
        cleared=True, delta_P_pu=delta_P, H_sys_s=H_sys, rocof_hz_s=rocof,
        nadir_hz=nadir, t_nadir_s=t_nadir, n_responsive=n_responsive,
        reason=f"RoCoF {rocof:.4f} Hz/s and nadir {nadir:.4f} Hz both clear")


def integrate_nadir(config: ScreenConfig,
                    network: Network,
                    dynamics: Dict[int, UnitDynamics],
                    live_gens: Sequence[int],
                    P_star: Dict[int, float],
                    delta_P: float,
                    H_sys: float) -> Tuple[float, float, int]:
    """Integrate the centre-of-inertia response.  Returns (nadir Hz, t, n).

    One COI mass on the system base,

        2 H_sys d(df)/dt = sum_g dPm_g - delta_P - D df

    with each responsive unit ramping toward its droop-implied target through a
    first-order governor lag and capped by its headroom,

        T_g d(dPm_g)/dt = clamp(-df / R_g, 0, headroom_g) - dPm_g.

    A unit with no governor record contributes no primary response, and one at
    its ceiling contributes no headroom.
    """
    if delta_P <= 0.0:
        # A surplus produces an over-frequency excursion, and the screen tests
        # the under-frequency nadir against the UFLS stage.  Nothing to reject.
        return config.f0_hz, 0.0, 0

    responsive: List[Tuple[float, float, float]] = []       # (R, T, headroom)
    for count in live_gens:
        unit = dynamics.get(count)
        if unit is None or not unit.is_responsive:
            continue
        headroom = network.gens[count].Pmax - float(P_star.get(count, 0.0))
        if headroom <= 0.0:
            continue
        responsive.append((float(unit.R), float(unit.T), headroom))

    damping = config.damping.on_system(network)
    dt = config.dt_s
    steps = max(1, int(round(config.horizon_s / dt)))
    df = 0.0                                     # p.u. frequency deviation
    pm = [0.0] * len(responsive)
    worst, t_worst = 0.0, 0.0

    for step in range(steps):
        total = 0.0
        for index, (droop, lag, headroom) in enumerate(responsive):
            target = -df / droop
            if target < 0.0:
                target = 0.0
            elif target > headroom:
                target = headroom
            pm[index] += dt * (target - pm[index]) / lag
            total += pm[index]
        df += dt * (total - delta_P - damping * df) / (2.0 * H_sys)
        if df < worst:
            worst, t_worst = df, (step + 1) * dt

    return config.f0_hz * (1.0 + worst), t_worst, len(responsive)
