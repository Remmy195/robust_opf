"""ropf.network against MATPOWER 8.1 (fixture written by fixtures/dump_matpower.m)."""

import math
import os

import pytest

from conftest import CASE, FIXTURES
from ropf.network import Branch, CaseFormatError, read_matpower

TOL = 1e-12


@pytest.fixture(scope="module")
def dump():
    meta, sections, header, current = {}, {}, None, None
    with open(os.path.join(FIXTURES, "matpower_ACTIVSg200.csv")) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# "):
                parts = line[2:].split(None, 1)
                if len(parts) == 2:
                    meta[parts[0]] = parts[1]
            elif line.startswith("SECTION "):
                current, header = line.split(None, 1)[1], None
                sections[current] = []
            elif header is None:
                header = line.split(",")
            else:
                sections[current].append(dict(zip(header, line.split(","))))
    return meta, sections


def close(actual, expected):
    return math.isclose(actual, float(expected), rel_tol=TOL, abs_tol=TOL)


def test_branch_admittance_and_dc_model(net, dump):
    for row in dump[1]["branch"]:
        br = net.branches[int(row["row_ext"])]
        assert (br.f, br.t) == (int(row["f_ext"]), int(row["t_ext"]))
        for name in ("Gff", "Bff", "Gft", "Bft", "Gtf", "Btf", "Gtt", "Btt",
                     "bdc", "Pfinj"):
            assert close(getattr(br, name), row[name]), (br.count, name)
        assert br.Ptinj == -br.Pfinj


def test_buses(net, dump):
    base = net.baseMVA
    for row in dump[1]["bus"]:
        bus = net.buses[int(row["row_ext"])]
        assert (bus.nodeID, bus.nodetype) == (int(row["id_ext"]), int(row["type"]))
        if bus.nodetype == 4:
            assert bus.Pd == bus.Qd == bus.Gs == bus.Bs == 0.0
            continue
        for name in ("Pd", "Qd", "Gs", "Bs"):
            assert close(getattr(bus, name), float(row[name]) / base)
        assert close(bus.Vmax, row["Vmax"]) and close(bus.Vmin, row["Vmin"])


def test_generators_and_costs(net, dump):
    base = net.baseMVA
    for row in dump[1]["gen"]:
        gen = net.gens[int(row["row_ext"])]
        assert gen.nodeID == int(row["bus_ext"])
        assert gen.status == (1 if int(row["status"]) > 0 else 0)
        for name in ("Pmax", "Pmin", "Qmax", "Qmin"):
            assert close(getattr(gen, name), float(row[name]) / base)
        expected = (float(row["c2"]) * base ** 2, float(row["c1"]) * base,
                    float(row["c0"]))
        assert all(close(a, b) for a, b in zip(gen.costvector, expected))


def _recost(tmp_path, cubic):
    with open(CASE) as handle:
        text = handle.read()
    text = text.replace("\t2\t0\t0\t3\t", f"\t2\t0\t0\t4\t{cubic}\t")
    path = tmp_path / "case_cubic.m"
    path.write_text(text)
    return str(path)


def test_cubic_form_with_zero_leading_term(net, tmp_path):
    cubic = read_matpower(_recost(tmp_path, "0"))
    for count, gen in net.gens.items():
        assert cubic.gens[count].costvector == gen.costvector


def test_genuinely_cubic_cost_is_an_error(tmp_path):
    with pytest.raises(CaseFormatError):
        read_matpower(_recost(tmp_path, "0.5"))


def test_zero_rate_a_becomes_big_m(net, dump):
    big_m = 2 * net.sumPd / net.baseMVA
    for row in dump[1]["branch"]:
        br = net.branches[int(row["row_ext"])]
        rate_a = float(row["rateA"]) / net.baseMVA
        assert br.constrainedflow == (rate_a != 0.0)
        assert close(br.limit, rate_a if rate_a else big_m)


def test_counts(net, dump):
    ext = dump[0]["ext_counts"].split()
    counts = {ext[i]: int(ext[i + 1]) for i in range(0, len(ext), 2)}
    assert (net.numbuses, net.numgens, net.branchcount) == \
        (counts["bus"], counts["gen"], counts["branch"])
    assert net.numbranches == len(dump[1]["branch"])


def test_negative_resistance_contributes_no_heat():
    br = Branch(1, 1, 1, 2, 2, r=-0.0045, x=0.01, bc=0.0, rateA=1.0, ratio=1.0,
                angle=0.0, minangle=-30.0, maxangle=30.0, defaultlimit=10.0)
    assert br.r == -0.0045 and br.r_heat == 0.0
