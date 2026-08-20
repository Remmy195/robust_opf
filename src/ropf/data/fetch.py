"""Download the ACTIVSg test systems into ``data/``.

The six rungs come from the Texas A&M electric grid test case repository.  The
cases are not committed -- the six distributions run to about 875 MB -- so a
fresh clone fetches them with::

    ropf fetch                # all six rungs
    ropf fetch activs200      # one of them
    ropf fetch --list

THE DISTRIBUTIONS ARE NOT UNIFORM, and the exceptions are not guessable from
the pattern the other rungs follow.  Two are recorded in `SOURCES` and handled
here so that nothing downstream has to know them:

    * ACTIVSg70k ships as an unpacked directory rather than a zip
    * ACTIVSg25k names its dynamics file ``ACTIVSg25k.dyr``, where the other
      five name theirs ``ACTIVSg<n>_dynamics.dyr``

The second is not this module's problem -- `ropf.counterfactual.dynamics.locate`
handles it at read time -- but it is recorded here because this is where the
layout of a distribution is described.

WHAT IS VERIFIED.  Each rung declares the SHA-256 of the MATPOWER case file it
must produce.  The ACTIVSg distributions are not versioned, so two downloads of
the same URL can differ; without the digest a study could be reproduced against
a different case and report it as the same one.  A mismatch is an error, and
the digest is what `ropf.results` records in every summary.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

#: Where the cases land: the repository's top-level ``data/``.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data")

BASE_URL = "https://electricgrids.engr.tamu.edu/wp-content/uploads/sites/2/"


def _noop(_message: str) -> None:
    return None


@dataclass(frozen=True)
class Source:
    """One rung of the ladder, and where its case file comes from."""

    rung: str
    #: The MATPOWER file this must produce, under `DATA_DIR`.
    case: str
    description: str
    url: str = ""
    #: SHA-256 of `case`.  Empty means unpinned, and `fetch` says so rather
    #: than passing an unverified file off as verified.
    sha256: str = ""
    #: Path of the case inside the archive, when the download is a zip.
    member: str = ""
    #: True when the distribution ships unpacked.  ACTIVSg70k, and only it.
    unpacked: bool = False


#: The six rungs.  URLs are the TAMU landing-page downloads; a rung whose URL
#: is empty must be placed in ``data/`` by hand, and `fetch` says which.
SOURCES: Dict[str, Source] = {
    "activs200": Source(
        rung="ACTIVSg200", case="case_ACTIVSg200.m",
        description="ACTIVSg200, the Illinois 200-bus synthetic grid",
        member="case_ACTIVSg200.m",
        sha256="3c92cb217e1e04bb764d2566ccf01f3f2e2ac8af2d6b2907b0619ee335165c87"),
    "activs500": Source(
        rung="ACTIVSg500", case="case_ACTIVSg500.m",
        description="ACTIVSg500, the South Carolina 500-bus synthetic grid",
        member="case_ACTIVSg500.m",
        sha256="8ca6d54ea5179eeb03fe29d7b645618e7a86338c172247e81687476660f6dcbe"),
    "texas2k": Source(
        rung="ACTIVSg2000", case="case_ACTIVSg2000.m",
        description="ACTIVSg2000, the Texas 2000-bus synthetic grid",
        member="case_ACTIVSg2000.m",
        sha256="5edb60e97153c27de68403499174557213090a5b04da1bbc34b3196715c5da94"),
    "activs10k": Source(
        rung="ACTIVSg10k", case="case_ACTIVSg10k.m",
        description="ACTIVSg10k, the Western US 10,000-bus synthetic grid",
        member="case_ACTIVSg10k.m",
        sha256="ead10b25fecc4dcc02f88bacdfb3526fe8b8985b81f7e539c95abddb32575590"),
    "activs25k": Source(
        rung="ACTIVSg25k", case="case_ACTIVSg25k.m",
        description="ACTIVSg25k, the Eastern US 25,000-bus synthetic grid "
                    "(names its dynamics file ACTIVSg25k.dyr, not "
                    "ACTIVSg25k_dynamics.dyr)",
        member="case_ACTIVSg25k.m",
        sha256="0b7c131ff6434491f5c0f76dedf67bff155d9cbb91ce67aef5ce275fd8bf3004"),
    "activs70k": Source(
        rung="ACTIVSg70k", case="case_ACTIVSg70k.m",
        description="ACTIVSg70k, the USA 70,000-bus synthetic grid "
                    "(ships unpacked as a directory, not as a zip)",
        member="ACTIVSg70k/case_ACTIVSg70k.m", unpacked=True,
        sha256="5df8c785c75f174555d307e05ae279c51f888ebbd85c469dab3265baf3e96293"),
}

#: The ladder, smallest first.  This is the order `ropf fetch` and the study
#: driver both use, so a partial run covers the cheap rungs first.
LADDER = ("activs200", "activs500", "texas2k", "activs10k", "activs25k",
          "activs70k")


###############################################################################
# Digests
###############################################################################


def digest(path: str) -> str:
    """SHA-256 of a file, streamed: the 70k case is 19 MB."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(source: Source, path: str,
           log: Optional[Callable[[str], None]] = None) -> bool:
    """Check a case against its declared digest.  Unpinned rungs say so."""
    emit = log or _noop
    if not source.sha256:
        emit(f"   {source.case}: no digest declared for this rung, so its "
             f"contents are NOT verified\n")
        return True
    found = digest(path)
    if found == source.sha256:
        emit(f"   {source.case}: sha256 matches\n")
        return True
    emit(f"   {source.case}: SHA-256 MISMATCH\n"
         f"     expected {source.sha256}\n"
         f"     found    {found}\n"
         f"   The ACTIVSg distributions are not versioned, so this is a "
         f"different file under the same name. Results produced against it "
         f"are not the results this study reports.\n")
    return False


###############################################################################
# Fetching
###############################################################################


def status(names: Optional[Sequence[str]] = None,
           data_dir: str = DATA_DIR) -> List[tuple]:
    """(name, source, present) for each requested rung.

    `data_dir` is a parameter and not the constant because `fetch` takes one
    too: a presence check against a different directory than the one being
    written reports a rung as present and then digests a file that is not
    there.
    """
    rows = []
    for name in (names or LADDER):
        if name not in SOURCES:
            raise KeyError(f"unknown rung {name!r}; known: "
                           f"{', '.join(sorted(SOURCES))}")
        source = SOURCES[name]
        rows.append((name, source,
                     os.path.isfile(os.path.join(data_dir, source.case))))
    return rows


def fetch(names: Optional[Sequence[str]] = None,
          log: Optional[Callable[[str], None]] = None,
          data_dir: str = DATA_DIR) -> int:
    """Download and unpack the requested rungs.  Returns a process exit code.

    A rung already present is verified and left alone, so re-running is cheap
    and is the way to check an existing tree.
    """
    emit = log or _noop
    os.makedirs(data_dir, exist_ok=True)
    missing_url, failed = [], []

    for name, source, present in status(names, data_dir):
        target = os.path.join(data_dir, source.case)
        emit(f"\n {name}: {source.description}\n")

        if present:
            emit(f"   already at {target}\n")
            if not verify(source, target, emit):
                failed.append(name)
            continue

        if not source.url:
            emit(f"   no download URL is recorded for this rung.\n")
            missing_url.append((name, source))
            continue

        try:
            _download_and_unpack(source, target, emit)
        except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
            emit(f"   download failed: {exc}\n")
            failed.append(name)
            continue
        if not verify(source, target, emit):
            failed.append(name)

    if missing_url:
        emit("\n" + "-" * 70 + "\n")
        emit("Some rungs have no recorded download URL. The Texas A&M\n"
             "repository serves them from a landing page that requires\n"
             "accepting terms, so the file cannot be retrieved unattended.\n"
             "Download them from\n\n"
             f"    {BASE_URL.rsplit('/wp-content', 1)[0]}/activsg-cases/\n\n"
             f"and place the MATPOWER file in {data_dir}:\n")
        for name, source in missing_url:
            emit(f"    {source.case:<24} for `{name}`\n")

    if failed:
        emit(f"\n {len(failed)} rung(s) could not be verified: "
             f"{', '.join(failed)}\n")
        return 1
    return 0


###############################################################################
# Adopting a local distribution
###############################################################################


def _members(source_dir: str, dist: str):
    """``(read, {basename: member})`` for a distribution, zip or unpacked.

    The two shapes are flattened here so that `adopt` never branches on which
    one it got.  `read` takes a member name and returns bytes.
    """
    archive = os.path.join(source_dir, f"{dist}.zip")
    if os.path.isfile(archive):
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
        index = {os.path.basename(n): n for n in names if not n.endswith("/")}

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

    ACTIVSg2000 ships its dynamics aux as ``.AUX`` where every other rung uses
    ``.aux``, and a case-sensitive lookup silently returns nothing.
    """
    lowered = {name.lower(): member for name, member in index.items()}
    for candidate in candidates:
        member = lowered.get(candidate.lower())
        if member is not None:
            return member
    return None


def adopt(source_dir: str,
          names: Optional[Sequence[str]] = None,
          log: Optional[Callable[[str], None]] = None,
          data_dir: str = DATA_DIR) -> int:
    """Take the cases AND the dynamics from local ACTIVSg distributions.

    `source_dir` holds the distributions as TAMU ships them -- ``ACTIVSg200.zip``
    and the like, or an unpacked directory of the same name.  For each rung this
    writes three files into `data_dir`:

        case_ACTIVSg<n>.m          the MATPOWER case, verified against its digest
        ACTIVSg<n>_dynamics.dyr    the machine records the frequency screen needs
        ACTIVSg<n>.aux             the GenParFac participation factors

    THIS IS THE PATH THAT ACTUALLY WORKS, and `fetch` is not.  Texas A&M serves
    the distributions from a terms-gated landing page, so no rung carries a URL
    and nothing can be retrieved unattended; whoever runs this study has already
    downloaded the archives by hand.  Adopting them is then a copy, not a
    download, and it is the only step that has ever been needed.

    It also takes the dynamics, which `fetch` never did.  Those ship INSIDE the
    same archives as the cases, so a tree built by downloading case files alone
    leaves the Section 4.2 screen with nothing to read, and the study config has
    to point at wherever the archives happened to be unpacked instead.  The
    names are the ones `ropf.counterfactual.dynamics.locate` looks for, so after
    this ``dynamics_search = data`` is correct and self-contained.
    """
    emit = log or _noop
    os.makedirs(data_dir, exist_ok=True)
    failed, missing = [], []

    for name, source, _present in status(names, data_dir):
        dist = source.rung
        emit(f"\n {name}: {source.description}\n")

        read, index, where = _members(source_dir, dist)
        if read is None:
            emit(f"   no {dist}.zip and no {dist}/ under {source_dir}\n")
            missing.append(name)
            continue
        emit(f"   from {where}\n")

        case_member = _pick(index, os.path.basename(source.member),
                            os.path.basename(source.case))
        if case_member is None:
            emit(f"   {source.case} is not in the distribution\n")
            failed.append(name)
            continue

        # VERIFIED BEFORE IT REPLACES ANYTHING.  The archives are not the only
        # place these cases come from, and they are not versioned: an archive
        # can hold a different vintage of case_ACTIVSg<n>.m under the same name
        # as the tree already has.  Writing first and checking after would let
        # one command silently swap the case a study was run against for a
        # different one, and the digest would only tell you afterwards.
        target = os.path.join(data_dir, source.case)
        payload = read(case_member)
        found = hashlib.sha256(payload).hexdigest()

        if source.sha256 and found != source.sha256:
            emit(f"   {source.case}: SHA-256 MISMATCH, NOT WRITTEN\n"
                 f"     expected {source.sha256}\n"
                 f"     found    {found}\n"
                 f"   The copy in this distribution is a different vintage of "
                 f"the case than the one this study is pinned to. The existing "
                 f"file was left alone.\n")
            failed.append(name)
        else:
            with open(target, "wb") as handle:
                handle.write(payload)
            emit(f"   wrote {source.case}"
                 f"{'' if source.sha256 else ' (unpinned, NOT verified)'}\n")

        # The dynamics.  Absent is reported, not fatal: the frontier runs
        # without them and only the Section 4.2 screen needs them.
        dyr = _pick(index, f"{dist}_dynamics.dyr", f"{dist}.dyr")
        if dyr is None:
            emit(f"   no .dyr in the distribution; the frequency screen will "
                 f"have no machine records for this rung\n")
        else:
            # Written under the name `locate` looks for first, so the ACTIVSg25k
            # irregularity stops being visible downstream.
            out = os.path.join(data_dir, f"{dist}_dynamics.dyr")
            with open(out, "wb") as handle:
                handle.write(read(dyr))
            emit(f"   wrote {os.path.basename(out)}\n")

        aux = _pick(index, f"{dist}.aux", f"{dist}_dynamics.aux")
        if aux is None:
            emit(f"   no .aux; participation falls back to the capacity "
                 f"surrogate for this rung\n")
        else:
            out = os.path.join(data_dir, f"{dist}.aux")
            with open(out, "wb") as handle:
                handle.write(read(aux))
            emit(f"   wrote {os.path.basename(out)}\n")

    if missing:
        emit(f"\n {len(missing)} distribution(s) were not found under "
             f"{source_dir}: {', '.join(missing)}\n")
    if failed:
        emit(f"\n {len(failed)} rung(s) could not be verified: "
             f"{', '.join(failed)}\n")
        return 1
    return 0


def _download_and_unpack(source: Source, target: str,
                         emit: Callable[[str], None]) -> None:
    emit(f"   downloading {source.url}\n")
    with tempfile.TemporaryDirectory() as workdir:
        archive = os.path.join(workdir, "download")
        with urllib.request.urlopen(source.url, timeout=300) as response, \
                open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as held:
                member = _find_member(held, source)
                emit(f"   extracting {member}\n")
                with held.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            # An unpacked distribution: the download is the case itself.
            shutil.copyfile(archive, target)
    emit(f"   wrote {target}\n")


def _find_member(archive: zipfile.ZipFile, source: Source) -> str:
    """The case inside the archive, tolerating a wrapping directory."""
    names = archive.namelist()
    if source.member in names:
        return source.member
    basename = os.path.basename(source.member)
    for name in names:
        if os.path.basename(name) == basename:
            return name
    raise FileNotFoundError(
        f"{source.case} is not in the archive; it holds {len(names)} entries, "
        f"none named {basename}")


if __name__ == "__main__":
    raise SystemExit(fetch(sys.argv[1:] or None, log=sys.stdout.write))
