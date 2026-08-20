"""`ropf.network` against MATPOWER 8.1.

This is the anchor test of the repository.  Every number the study reports rests
on the network model being MATPOWER's, so that claim is checked directly rather
than asserted in a comment.

The fixtures under ``tests/fixtures/matpower_*.csv`` are produced by
``dump_matpower.m``, which reads the quantities out of MATPOWER's own assembled
``makeYbus`` and ``makeBdc`` matrices.  Nothing in the generation path imports
``ropf``; regenerating the fixture from the code under test would make this
circular and prove nothing.

To regenerate::

    OCTAVE_HOME=$HOME/miniconda3 $OCTAVE_HOME/bin/octave-cli --no-gui \
        tests/fixtures/dump_matpower.m /path/to/case_ACTIVSg200.m \
        tests/fixtures/matpower_ACTIVSg200.csv

Two deliberate deviations from MATPOWER are asserted here as deviations, so that
they cannot be silently "fixed" back: the ``rateA == 0`` big-M substitution, and
the zeroing of demand at isolated buses.
"""

from __future__ import annotations

import math
import os

import pytest

from ropf.network import read_matpower

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

#: MATPOWER writes %.17g, so agreement is limited by double rounding in the
#: decimal round trip, not by the formulas.  The recorded worst case over these
#: cases is ~4e-15 absolute on the admittance entries.
ATOL = 1e-12
RTOL = 1e-12


def _load_fixture(name):
    """Parse a dump into ``(meta, {section: [row dicts]})``."""
    path = os.path.join(FIXTURES, name)
    meta, sections, header, current = {}, {}, None, None
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# "):
                parts = line[2:].split(None, 1)
                if len(parts) == 2:
                    meta[parts[0]] = parts[1]
                continue
            if line.startswith("SECTION "):
                current = line.split(None, 1)[1]
                sections[current] = []
                header = None
                continue
            if header is None:
                header = line.split(",")
                continue
            values = line.split(",")
            sections[current].append(dict(zip(header, values)))
    return meta, sections


def _close(actual, expected, what):
    assert math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL), (
        f"{what}: got {actual!r}, MATPOWER says {expected!r}, "
        f"difference {actual - expected:g}")


def _case_path(basename):
    """Prefer the committed fixture case, then the fetched data tree.

    No absolute path to any particular machine appears here.  A larger case is
    found only if the distribution has been unpacked into ``data/``, or if the caller
    points ``ROPF_DATA_DIR`` at a tree that has it.
    """
    roots = [FIXTURES,
             os.path.join(os.path.dirname(__file__), os.pardir, "data")]
    env_root = os.environ.get("ROPF_DATA_DIR")
    if env_root:
        roots.append(env_root)
    for root in roots:
        candidate = os.path.join(root, basename)
        if os.path.exists(candidate):
            return candidate
    return None


CASES = [
    # (fixture, case file, is the case committed with the repo?)
    ("matpower_ACTIVSg200.csv", "case_ACTIVSg200.m", True),
    ("matpower_ACTIVSg2000.csv", "case_ACTIVSg2000.m", False),
]


@pytest.fixture(scope="module", params=CASES, ids=[c[1] for c in CASES])
def parity_case(request):
    fixture_name, case_basename, committed = request.param
    fixture_path = os.path.join(FIXTURES, fixture_name)
    if not os.path.exists(fixture_path):
        pytest.skip(f"fixture {fixture_name} not generated")
    case_path = _case_path(case_basename)
    if case_path is None:
        if committed:
            pytest.fail(f"{case_basename} should ship with the repo")
        pytest.skip(f"{case_basename} not available; unpack it into data/")
    meta, sections = _load_fixture(fixture_name)
    return meta, sections, read_matpower(case_path)


###############################################################################
# Admittance and DC model
###############################################################################


def test_branch_admittance_matches_makeybus(parity_case):
    """The eight Y entries per branch, straight out of MATPOWER's Yf and Yt."""
    _, sections, net = parity_case
    rows = sections["branch"]
    assert rows, "fixture carries no branches"

    for row in rows:
        count = int(row["row_ext"])
        assert count in net.branches, (
            f"branch row {count} is in service for MATPOWER but absent here")
        br = net.branches[count]

        # The fixture records external bus IDs; the reader keys by count.
        assert br.f == int(row["f_ext"])
        assert br.t == int(row["t_ext"])

        for name in ("Gff", "Bff", "Gft", "Bft", "Gtf", "Btf", "Gtt", "Btt"):
            _close(getattr(br, name), float(row[name]),
                   f"branch {count} {name}")


def test_branch_dc_model_matches_makebdc(parity_case):
    """``bdc`` and the phase-shift injection, out of MATPOWER's Bf and Pfinj."""
    _, sections, net = parity_case
    for row in sections["branch"]:
        br = net.branches[int(row["row_ext"])]
        _close(br.bdc, float(row["bdc"]), f"branch {br.count} bdc")
        _close(br.Pfinj, float(row["Pfinj"]), f"branch {br.count} Pfinj")


def test_dc_injections_antisymmetric(parity_case):
    """``Ptinj == -Pfinj`` exactly, phase shifters included.

    The post-event model (D) carries only the from-end flow and takes
    ``Pt == -Pf``.  That identity holds because the two injections cancel; if
    this ever fails, ``postevent.mod`` is silently wrong on any case with a
    phase shifter.
    """
    _, _, net = parity_case
    for br in net.branches.values():
        assert br.Ptinj == -br.Pfinj, f"branch {br.count} injections not antisymmetric"


###############################################################################
# Buses and generators
###############################################################################


def test_bus_quantities_match(parity_case):
    """Demand, shunts and voltage limits, with the isolated-bus deviation."""
    _, sections, net = parity_case
    base = net.baseMVA
    n_isolated = 0

    for row in sections["bus"]:
        count = int(row["row_ext"])
        bus = net.buses[count]
        assert bus.nodeID == int(row["id_ext"])
        assert bus.nodetype == int(row["type"])

        if bus.nodetype == 4:
            # DELIBERATE DEVIATION.  MATPOWER's ext2int deletes isolated buses;
            # we keep them so the AMPL index set stays contiguous, and zero what
            # they would contribute.  A nonzero Pd at a bus with no incident
            # branch makes its own balance row infeasible.
            n_isolated += 1
            assert bus.Pd == 0.0 and bus.Qd == 0.0
            assert bus.Gs == 0.0 and bus.Bs == 0.0
            continue

        _close(bus.Pd, float(row["Pd"]) / base, f"bus {count} Pd")
        _close(bus.Qd, float(row["Qd"]) / base, f"bus {count} Qd")
        _close(bus.Gs, float(row["Gs"]) / base, f"bus {count} Gs")
        _close(bus.Bs, float(row["Bs"]) / base, f"bus {count} Bs")
        _close(bus.Vmax, float(row["Vmax"]), f"bus {count} Vmax")
        _close(bus.Vmin, float(row["Vmin"]), f"bus {count} Vmin")

    assert n_isolated == net.numisolated


def test_generator_limits_match(parity_case):
    """Output limits, converted to p.u., including out-of-service units."""
    _, sections, net = parity_case
    base = net.baseMVA
    for row in sections["gen"]:
        count = int(row["row_ext"])
        gen = net.gens[count]
        assert gen.nodeID == int(row["bus_ext"])
        # MATPOWER's status is an arbitrary integer; the reader normalizes to 0/1.
        assert gen.status == (1 if int(row["status"]) > 0 else 0)
        _close(gen.Pmax, float(row["Pmax"]) / base, f"gen {count} Pmax")
        _close(gen.Pmin, float(row["Pmin"]) / base, f"gen {count} Pmin")
        _close(gen.Qmax, float(row["Qmax"]) / base, f"gen {count} Qmax")
        _close(gen.Qmin, float(row["Qmin"]) / base, f"gen {count} Qmin")


def test_cost_coefficients_rescaled_to_pu(parity_case):
    """``c_k`` is per MW in the file and multiplies ``Pg**k`` in p.u. downstream.

    The rescale is ``c_k * baseMVA**k``, so the quadratic term moves by
    ``baseMVA**2`` -- a factor of 10,000 at the default base.  Getting this
    wrong is not subtle in the objective, but it is silent in the dispatch.
    """
    _, sections, net = parity_case
    base = net.baseMVA
    for row in sections["gen"]:
        count = int(row["row_ext"])
        gen = net.gens[count]
        expected = [float(row["c2"]) * base ** 2,
                    float(row["c1"]) * base,
                    float(row["c0"])]
        for slot, want in enumerate(expected):
            _close(gen.costvector[slot], want, f"gen {count} costvector[{slot}]")


###############################################################################
# The rateA deviation
###############################################################################


def test_rate_a_zero_becomes_big_m(parity_case):
    """``rateA == 0`` is the one substitution that is not MATPOWER.

    MATPOWER reads a zero rating as "unlimited" and leaves the branch
    unconstrained.  AMPL needs a finite bound on ``Pf``, so a zero rating
    becomes ``2 * sum(Pd) / baseMVA``.  The substitution must be:

      * applied exactly when MATPOWER's rateA is zero,
      * flagged by ``constrainedflow == 0`` so the count is auditable, and
      * non-binding, i.e. far above any flow the network can carry.
    """
    _, sections, net = parity_case
    base = net.baseMVA
    big_m = 2 * net.sumPd / base
    n_substituted = 0

    for row in sections["branch"]:
        br = net.branches[int(row["row_ext"])]
        rate_a = float(row["rateA"])
        if rate_a == 0.0:
            n_substituted += 1
            assert br.constrainedflow == 0
            _close(br.limit, big_m, f"branch {br.count} big-M limit")
        else:
            assert br.constrainedflow == 1
            _close(br.limit, rate_a / base, f"branch {br.count} limit")

    assert n_substituted == len(net.unconstrained_branches())
    # Non-binding by construction: twice the total system load cannot flow on a
    # single line.  If this ever fails the substitution has started to bind and
    # the reported flows are shaped by an artificial bound.
    if n_substituted:
        assert big_m > 2 * max(
            br.limit for br in net.branches.values() if br.constrainedflow)


###############################################################################
# Counts
###############################################################################


def test_element_counts_match_fixture(parity_case):
    meta, sections, net = parity_case
    ext = meta["ext_counts"].split()
    counts = {ext[i]: int(ext[i + 1]) for i in range(0, len(ext), 2)}
    assert net.numbuses == counts["bus"]
    assert net.numgens == counts["gen"]
    # `branchcount` is every row read; `numbranches` holds only in-service ones,
    # which is what the fixture's branch section carries.
    assert net.branchcount == counts["branch"]
    assert net.numbranches == len(sections["branch"])
