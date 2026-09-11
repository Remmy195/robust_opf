# ropf

Risk-aware optimal power flow by cutting planes. Companion code for the
manuscript *Risk-aware optimal power flow*.

## Requirements

- Python 3.10 or later
- AMPL with a valid licence
- Gurobi, plus Knitro or Ipopt for the AC stages (`a2`, `a3`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,study]"
```

## Case data

The ACTIVSg cases are not included. Download them from the Texas A&M Electric
Grid Test Case Repository and unpack them into `data/`:

```bash
cd data
for z in /path/to/ACTIVSg*.zip; do
    unzip -j -o "$z" 'case_ACTIVSg*.m' '*.dyr' '*.aux'
done
```

ACTIVSg70k comes as a folder; copy its `.m`, `.dyr` and `.aux` files into
`data/` by hand.

## Run

```bash
ropf solve configs/activs200_flow_baseline.conf   # one case, no download needed
ropf ladder configs/ladder.conf --all             # full study
ropf ladder configs/ladder.conf --status          # progress
```

Each command reads its settings from the config file. `ropf keys` lists every
key, `ropf --help` lists the commands, and `--dry-run` checks a config without
solving. `ropf ladder` can run in several shells at once, and rerunning it
resumes after a crash.

## Tests

```bash
pytest
```

## Licence

See [LICENSE](LICENSE).
