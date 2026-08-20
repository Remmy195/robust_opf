"""The six rungs, and installing them from the TAMU distributions.

The ACTIVSg cases are not committed -- the distributions run to about 875 MB --
so a fresh clone installs them from wherever they were downloaded:

    ropf data ~/ACTIVSg          # all six rungs
    ropf data ~/ACTIVSg texas2k  # one of them

THERE IS NO DOWNLOADER.  Texas A&M serves these from a landing page that
requires accepting terms, so nothing can be retrieved unattended and whoever
runs this study already has the archives.  Installing them is a copy.

WHAT IS COPIED, per rung: the MATPOWER case, the ``.dyr`` machine records the
Section 4.2 frequency screen reads, and the ``.aux`` participation factors.
All three live in the SAME distribution, which is the reason this exists rather
than a line of documentation saying `unzip`: taking only the case leaves the
screen with nothing to read.

THE DISTRIBUTIONS ARE NOT UNIFORM, and neither exception is guessable:

    * ACTIVSg70k ships as an unpacked directory rather than a zip
    * ACTIVSg25k names its dynamics file ``ACTIVSg25k.dyr`` where the other five
      use ``ACTIVSg<n>_dynamics.dyr``

Both are absorbed here -- the ``.dyr`` is written back under the name the other
five carry -- so nothing downstream has to know about either.

There is no digest table.  `ropf.results.provenance` records the SHA-256 of the
case each run actually used, in that run's own summary, which is the claim that
matters: it says what produced a number rather than what was supposed to.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Where the cases land: the repository's top-level ``data/``.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data")


def _noop(_message: str) -> None:
    return None


@dataclass(frozen=True)
class Rung:
    """One rung: what the distribution is called, and what case it holds."""

    #: The TAMU distribution name, e.g. ``ACTIVSg2000``.  Not the rung name.
    dist: str
    case: str
    description: str


#: The six rungs, smallest first.  This order is the one a partial run follows,
#: so the cheap rungs are covered first.
RUNGS: Dict[str, Rung] = {
    "activs200": Rung("ACTIVSg200", "case_ACTIVSg200.m",
                      "the Illinois 200-bus synthetic grid"),
    "activs500": Rung("ACTIVSg500", "case_ACTIVSg500.m",
                      "the South Carolina 500-bus synthetic grid"),
    "texas2k": Rung("ACTIVSg2000", "case_ACTIVSg2000.m",
                    "the Texas 2000-bus synthetic grid"),
    "activs10k": Rung("ACTIVSg10k", "case_ACTIVSg10k.m",
                      "the Western US 10,000-bus synthetic grid"),
    "activs25k": Rung("ACTIVSg25k", "case_ACTIVSg25k.m",
                      "the Eastern US 25,000-bus synthetic grid"),
    "activs70k": Rung("ACTIVSg70k", "case_ACTIVSg70k.m",
                      "the USA 70,000-bus synthetic grid"),
}

LADDER: Tuple[str, ...] = tuple(RUNGS)


def case_path(rung: str, data_dir: str = DATA_DIR) -> str:
    return os.path.join(data_dir, RUNGS[rung].case)


def _selected(names: Optional[Sequence[str]]) -> List[str]:
    for name in names or ():
        if name not in RUNGS:
            raise KeyError(f"unknown rung {name!r}; known: "
                           f"{', '.join(LADDER)}")
    return list(names or LADDER)


def _open(source_dir: str, dist: str):
    """``(read, {basename: member}, where)`` for a zip or an unpacked directory."""
    archive = os.path.join(source_dir, f"{dist}.zip")
    if os.path.isfile(archive):
        with zipfile.ZipFile(archive) as handle:
            index = {os.path.basename(n): n for n in handle.namelist()
                     if not n.endswith("/")}

        def read(member: str) -> bytes:
            with zipfile.ZipFile(archive) as handle:
                return handle.read(member)
        return read, index, archive

    directory = os.path.join(source_dir, dist)
    if os.path.isdir(directory):
        index = {name: os.path.join(directory, name)
                 for name in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, name))}

        def read(member: str) -> bytes:
            with open(member, "rb") as handle:
                return handle.read()
        return read, index, directory

    return None, {}, ""


def _pick(index: Dict[str, str], *candidates: str) -> Optional[str]:
    """The first candidate present, matched case-insensitively.

    ACTIVSg2000 ships its dynamics aux as ``.AUX`` where the others use
    ``.aux``, and a case-sensitive lookup silently finds nothing.
    """
    lowered = {name.lower(): member for name, member in index.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def install(source_dir: str,
            names: Optional[Sequence[str]] = None,
            log: Optional[Callable[[str], None]] = None,
            data_dir: str = DATA_DIR) -> int:
    """Copy the case and dynamics for each rung into `data_dir`.

    Returns a process exit code.  A distribution that is not in `source_dir` is
    reported and skipped rather than failing the run, so installing a subset is
    a normal thing to do.
    """
    emit = log or _noop
    os.makedirs(data_dir, exist_ok=True)
    missing = []

    for name in _selected(names):
        rung = RUNGS[name]
        emit(f"\n {name}: {rung.description}\n")

        read, index, where = _open(source_dir, rung.dist)
        if read is None:
            emit(f"   no {rung.dist}.zip and no {rung.dist}/ under "
                 f"{source_dir}\n")
            missing.append(name)
            continue
        emit(f"   from {where}\n")

        wanted = [
            (_pick(index, rung.case), rung.case),
            (_pick(index, f"{rung.dist}_dynamics.dyr", f"{rung.dist}.dyr"),
             f"{rung.dist}_dynamics.dyr"),
            (_pick(index, f"{rung.dist}.aux", f"{rung.dist}_dynamics.aux"),
             f"{rung.dist}.aux"),
        ]
        for member, target_name in wanted:
            if member is None:
                emit(f"   {target_name} is not in the distribution\n")
                continue
            target = os.path.join(data_dir, target_name)
            with open(target, "wb") as handle:
                handle.write(read(member))
            emit(f"   wrote {target_name}\n")

    if missing:
        emit(f"\n {len(missing)} distribution(s) not found under {source_dir}: "
             f"{', '.join(missing)}\n")
        return 1
    return 0
