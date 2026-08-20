"""The case fetcher: what it verifies, and what it refuses to claim.

The fetcher's job is not really downloading -- five of the six rungs are behind
a landing page that requires accepting terms, so they arrive by hand.  Its job
is to say, of a tree that already exists, whether it holds the cases this study
was run against.  That makes the digest the module's whole point, and these
tests are mostly about the digest.
"""

from __future__ import annotations

import os
import zipfile

import pytest

from ropf.data import fetch
from ropf.data.fetch import LADDER, SOURCES, Source

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CASE = os.path.join(FIXTURES, "case_ACTIVSg200.m")


def log():
    """A collecting log, so a test can assert on what the user was told."""
    lines = []
    return lines, lines.append


###############################################################################
# The rung table
###############################################################################


def test_the_ladder_and_the_source_table_agree():
    """A rung in one and not the other is a rung nothing can run."""
    assert set(LADDER) == set(SOURCES)
    assert len(LADDER) == len(SOURCES) == 6


def test_every_rung_is_pinned():
    """An unpinned rung silently downgrades `verify` to a no-op.

    The module is allowed to express "not verified" -- `verify` says so rather
    than lying -- but no rung of this study may actually be in that state, or a
    reproduction could run against a different case and report it as the same.
    """
    unpinned = [name for name, s in SOURCES.items() if not s.sha256]
    assert not unpinned, f"these rungs declare no digest: {unpinned}"
    for name, source in SOURCES.items():
        assert len(source.sha256) == 64, f"{name}: not a SHA-256"
        assert source.sha256 == source.sha256.lower().strip()


def test_the_declared_digests_are_distinct():
    """Two rungs sharing a digest means one was pasted over the other."""
    digests = [s.sha256 for s in SOURCES.values()]
    assert len(set(digests)) == len(digests)


def test_the_ladder_is_ordered_smallest_first():
    """A partial run must cover the cheap rungs first; the study driver
    relies on this order, not on the dict's insertion order."""
    assert LADDER[0] == "activs200"
    assert LADDER[-1] == "activs70k"


###############################################################################
# Digests
###############################################################################


def test_digest_matches_hashlib(tmp_path):
    import hashlib
    blob = tmp_path / "blob"
    payload = os.urandom(3 << 20)  # larger than the 1 MiB read chunk
    blob.write_bytes(payload)
    assert fetch.digest(str(blob)) == hashlib.sha256(payload).hexdigest()


def test_the_tracked_fixture_matches_the_declared_digest():
    """tests/fixtures/case_ACTIVSg200.m is the one case in the repository.

    It is the same file the fetcher would place in data/, so the digest pinned
    for activs200 must describe it. If this fails, the fixture and the study's
    smallest rung have drifted apart.
    """
    assert fetch.digest(CASE) == SOURCES["activs200"].sha256


def test_verify_accepts_a_matching_file():
    lines, emit = log()
    assert fetch.verify(SOURCES["activs200"], CASE, emit) is True
    assert "sha256 matches" in "".join(lines)


def test_verify_rejects_a_mismatch_and_prints_both_digests(tmp_path):
    """The message has to carry the found digest, or there is no way to tell a
    stale download from a corrupted one without rerunning sha256sum by hand."""
    impostor = tmp_path / "case_ACTIVSg200.m"
    impostor.write_text("function mpc = case_ACTIVSg200\n")
    lines, emit = log()
    assert fetch.verify(SOURCES["activs200"], str(impostor), emit) is False
    message = "".join(lines)
    assert "SHA-256 MISMATCH" in message
    assert SOURCES["activs200"].sha256 in message
    assert fetch.digest(str(impostor)) in message


def test_an_unpinned_rung_says_it_is_unverified():
    """`verify` may not return True quietly on a rung it cannot check."""
    lines, emit = log()
    loose = Source(rung="X", case="case_X.m", description="", sha256="")
    assert fetch.verify(loose, CASE, emit) is True
    assert "NOT verified" in "".join(lines)


###############################################################################
# status and fetch
###############################################################################


def test_status_rejects_an_unknown_rung():
    with pytest.raises(KeyError, match="unknown rung"):
        fetch.status(["activs300"])


def test_status_reports_presence_against_the_data_directory():
    rows = fetch.status(["activs200"])
    assert len(rows) == 1
    name, source, present = rows[0]
    assert name == "activs200"
    assert present == os.path.isfile(
        os.path.join(fetch.DATA_DIR, source.case))


def test_a_present_and_matching_case_is_left_alone(tmp_path):
    """Re-running the fetcher over a good tree is the way to check it, so it
    must verify in place and not re-download."""
    import shutil
    shutil.copyfile(CASE, tmp_path / "case_ACTIVSg200.m")
    lines, emit = log()
    code = fetch.fetch(["activs200"], log=emit, data_dir=str(tmp_path))
    message = "".join(lines)
    assert code == 0
    assert "already at" in message and "sha256 matches" in message
    assert "downloading" not in message


def test_a_present_but_wrong_case_fails_the_run(tmp_path):
    (tmp_path / "case_ACTIVSg200.m").write_text("not the case file\n")
    lines, emit = log()
    code = fetch.fetch(["activs200"], log=emit, data_dir=str(tmp_path))
    assert code == 1, "a digest mismatch must be a nonzero exit"
    assert "could not be verified" in "".join(lines)


def test_a_rung_with_no_url_explains_how_to_get_it(tmp_path):
    """No rung carries a URL: TAMU serves them from a terms-gated landing page.

    The fetcher must therefore not report success for a case it does not have.
    It has to name the file and where to put it.
    """
    assert all(not s.url for s in SOURCES.values()), (
        "a URL was added; this test describes the unattended-download case "
        "and needs revisiting")
    lines, emit = log()
    code = fetch.fetch(["texas2k"], log=emit, data_dir=str(tmp_path))
    message = "".join(lines)
    assert code == 0, "a missing case is not a verification failure"
    assert "no download URL" in message
    assert "case_ACTIVSg2000.m" in message
    assert str(tmp_path) in message


###############################################################################
# Archive layout
###############################################################################


def test_the_case_is_found_under_a_wrapping_directory(tmp_path):
    """The ACTIVSg zips are not uniform about a top-level folder."""
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("ACTIVSg200/case_ACTIVSg200.m", "x")
    with zipfile.ZipFile(archive) as z:
        assert (fetch._find_member(z, SOURCES["activs200"])
                == "ACTIVSg200/case_ACTIVSg200.m")


def test_a_missing_member_names_what_was_looked_for(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("readme.txt", "x")
    with zipfile.ZipFile(archive) as z:
        with pytest.raises(FileNotFoundError, match="case_ACTIVSg200.m"):
            fetch._find_member(z, SOURCES["activs200"])


###############################################################################
# Adopting a local distribution
###############################################################################


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    return str(path)


def test_adopt_takes_the_case_and_both_dynamics_files(tmp_path):
    """The .dyr and .aux ship inside the same archive as the case.

    Taking only the case is what left the frequency screen with nothing to read
    and forced the study config to point outside the repository.
    """
    source = tmp_path / "src"
    source.mkdir()
    data = tmp_path / "data"
    make_zip(source / "ACTIVSg200.zip", {
        "case_ACTIVSg200.m": open(CASE, "rb").read(),
        "ACTIVSg200_dynamics.dyr": "dyr payload",
        "ACTIVSg200.aux": "aux payload",
        "ACTIVSg200.RAW": "ignored",
    })
    lines, emit = log()
    code = fetch.adopt(str(source), ["activs200"], log=emit,
                       data_dir=str(data))
    assert code == 0, "".join(lines)
    assert (data / "case_ACTIVSg200.m").exists()
    assert (data / "ACTIVSg200_dynamics.dyr").read_text() == "dyr payload"
    assert (data / "ACTIVSg200.aux").read_text() == "aux payload"


def test_a_mismatched_case_is_not_written(tmp_path):
    """THE POINT OF CHECKING BEFORE WRITING.

    The archives are not versioned, so one can hold a different vintage of the
    case under the same name. Adopting it must not overwrite the case an
    existing tree was built on -- the digest would then only report the swap
    after it had happened.
    """
    source = tmp_path / "src"
    source.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    good = (data / "case_ACTIVSg200.m")
    good.write_bytes(open(CASE, "rb").read())

    make_zip(source / "ACTIVSg200.zip",
             {"case_ACTIVSg200.m": "a different vintage\n"})

    lines, emit = log()
    code = fetch.adopt(str(source), ["activs200"], log=emit,
                       data_dir=str(data))
    message = "".join(lines)
    assert code == 1
    assert "NOT WRITTEN" in message
    assert fetch.digest(str(good)) == SOURCES["activs200"].sha256, \
        "the existing verified case must be left exactly as it was"


def test_adopt_normalizes_the_activs25k_dyr_name(tmp_path):
    """ACTIVSg25k drops the `_dynamics` the other five carry.

    It is written back under the name `locate` looks for first, so the
    irregularity stops being visible to anything downstream.
    """
    source = tmp_path / "src"
    source.mkdir()
    data = tmp_path / "data"
    make_zip(source / "ACTIVSg25k.zip", {
        "case_ACTIVSg25k.m": "not the real case",
        "ACTIVSg25k.dyr": "dyr payload",
    })
    lines, emit = log()
    fetch.adopt(str(source), ["activs25k"], log=emit, data_dir=str(data))
    assert (data / "ACTIVSg25k_dynamics.dyr").read_text() == "dyr payload"


def test_adopt_reads_an_unpacked_distribution(tmp_path):
    """ACTIVSg70k ships as a directory rather than a zip."""
    source = tmp_path / "src"
    unpacked = source / "ACTIVSg70k"
    unpacked.mkdir(parents=True)
    (unpacked / "case_ACTIVSg70k.m").write_text("not the real case")
    (unpacked / "ACTIVSg70k_dynamics.dyr").write_text("dyr payload")
    data = tmp_path / "data"

    lines, emit = log()
    fetch.adopt(str(source), ["activs70k"], log=emit, data_dir=str(data))
    assert (data / "ACTIVSg70k_dynamics.dyr").read_text() == "dyr payload"


def test_adopt_finds_a_case_insensitive_aux(tmp_path):
    """ACTIVSg2000 ships its dynamics aux as .AUX where the others use .aux."""
    source = tmp_path / "src"
    source.mkdir()
    data = tmp_path / "data"
    make_zip(source / "ACTIVSg2000.zip", {
        "case_ACTIVSg2000.m": "not the real case",
        "ACTIVSg2000_dynamics.AUX": "aux payload",
    })
    lines, emit = log()
    fetch.adopt(str(source), ["texas2k"], log=emit, data_dir=str(data))
    assert (data / "ACTIVSg2000.aux").read_text() == "aux payload"


def test_a_missing_distribution_is_reported_not_crashed(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    lines, emit = log()
    code = fetch.adopt(str(source), ["activs200"], log=emit,
                       data_dir=str(tmp_path / "data"))
    message = "".join(lines)
    assert code == 0, "a distribution that is absent is not a failed digest"
    assert "no ACTIVSg200.zip" in message
