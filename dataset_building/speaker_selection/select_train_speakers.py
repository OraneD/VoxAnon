
import argparse
import sys
from collections import defaultdict

import pandas as pd
import pulp
import yaml



def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = ["metadata", "output", "n_speakers", "balance"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Missing keys in config: {missing}")

    cfg.setdefault("id_column", "spk_id")
    cfg.setdefault("relabel", {})
    cfg.setdefault("filters", {})
    cfg.setdefault("targets", {})
    cfg.setdefault("weights", {})
    cfg.setdefault("solver", {})
    cfg.setdefault("count_values", {})
    return cfg


def apply_relabel(df, relabel):
    for col, mapping in relabel.items():
        col = col.lower()
        mapping = {str(k): str(v) for k, v in mapping.items()}
        known = set(mapping) | set(mapping.values())
        unknown = set(df[col].dropna().astype(str)) - known
        if unknown:
            print(f"WARNING: [{col}] values not in relabel map, left untouched: "
                  f"{sorted(unknown)}", file=sys.stderr)
        df[col] = df[col].map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
    return df


def apply_filters(df, filters):
    for col, values in (filters.get("keep") or {}).items():
        vals = [str(v).lower() for v in values]
        df = df[df[col.lower()].astype(str).str.lower().isin(vals)]
    for col, values in (filters.get("drop") or {}).items():
        vals = [str(v).lower() for v in values]
        df = df[~df[col.lower()].astype(str).str.lower().isin(vals)]
    return df


def load_metadata(cfg):
    df = pd.read_csv(cfg["metadata"])
    df.columns = [c.lower() for c in df.columns]
    id_col = cfg["id_column"].lower()

    df = apply_relabel(df, cfg["relabel"])
    df = apply_filters(df, cfg["filters"])

    for col, values in cfg["balance"].items():
        df = df[df[col.lower()].astype(str).isin([str(v) for v in values])]

    df = df.dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    return df.reset_index(drop=True)


def even_split(n, values):
    base, rem = divmod(n, len(values))
    return {str(v): base + (1 if i < rem else 0) for i, v in enumerate(values)}


def build_targets(cfg):
    n = cfg["n_speakers"]
    targets = {}
    for col, values in cfg["balance"].items():
        explicit = cfg["targets"].get(col)
        targets[col.lower()] = (
            {str(k): v for k, v in explicit.items()}
            if explicit else even_split(n, values)
        )
        total = sum(targets[col.lower()].values())
        if total != n:
            print(f"WARNING: targets for '{col}' sum to {total}, expected {n}.",
                  file=sys.stderr)
    return targets


def optimize_selection(data, n, targets, weights, id_col, solver_cfg):
    if n > len(data):
        sys.exit(f"Cannot select {n} speakers out of {len(data)} candidates.")

    prob = pulp.LpProblem("speaker_selection", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(data))]
    prob += pulp.lpSum(x) == n  # exactly n speakers

    errors = []
    for col, target_dict in targets.items():
        rows_by_value = defaultdict(list)
        for i, v in enumerate(data[col].astype(str)):
            rows_by_value[v].append(i)

        weight = weights.get(col, 1)
        for value, target in target_dict.items():
            available = len(rows_by_value.get(str(value), []))
            if available < target:
                print(f"WARNING: only {available} candidates with {col} == {value} "
                      f"(target {target}).", file=sys.stderr)
            count = pulp.lpSum(x[i] for i in rows_by_value.get(str(value), []))
            err_pos = pulp.LpVariable(f"err_pos_{col}_{value}", lowBound=0)
            err_neg = pulp.LpVariable(f"err_neg_{col}_{value}", lowBound=0)
            prob += count - target == err_pos - err_neg
            errors.append(weight * (err_pos + err_neg))

    prob.setObjective(pulp.lpSum(errors))
    solver = pulp.PULP_CBC_CMD(
        msg=bool(solver_cfg.get("verbose", False)),
        timeLimit=solver_cfg.get("time_limit"),
    )
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        print(f"WARNING: solver status = {status}", file=sys.stderr)

    ids = data[id_col].tolist()
    return [ids[i] for i in range(len(data)) if pulp.value(x[i]) > 0.5]



def report(df, targets, count_values):
    print(f"\n=== SELECTION ({len(df)} speakers) ===")
    for col, target_dict in targets.items():
        counts = df[col].astype(str).value_counts()
        print(f"\n{col}:  value  actual / target")
        for value, target in target_dict.items():
            print(f"  {value:>12}  {counts.get(str(value), 0):>4} / {target}")

    for col, values in count_values.items():
        series = df[col.lower()].astype(str).str.lower()
        for value in values:
            n = int((series == str(value).lower()).sum())
            print(f"\n{col} == {value}: {n} / {len(df)} speakers")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    id_col = cfg["id_column"].lower()
    targets = build_targets(cfg)
    weights = {k.lower(): v for k, v in cfg["weights"].items()}

    df = load_metadata(cfg)
    print(f"{len(df)} candidate speakers after filtering.")

    print(f"Optimising selection of {cfg['n_speakers']} speakers...")
    ids = optimize_selection(df, cfg["n_speakers"], targets, weights, id_col, cfg["solver"])
    selected = df[df[id_col].isin(ids)].copy()

    selected.to_csv(cfg["output"], index=False)
    report(selected, targets, cfg["count_values"])
    print(f"\nSaved: {cfg['output']}")


if __name__ == "__main__":
    main()