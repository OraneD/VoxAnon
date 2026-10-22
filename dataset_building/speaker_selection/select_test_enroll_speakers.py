#!/usr/bin/env python
"""Select balanced enroll / trial speaker sets with an ILP (PuLP + CBC).

All parameters (input metadata, sizes, filters, targets, weights,
output paths) are read from a YAML file.

"""

import argparse
import sys
from collections import defaultdict

import pandas as pd
import pulp
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = ["metadata", "n_enroll", "n_trial", "output", "targets"]
    missing = [k for k in required if k not in cfg]
    if missing:
        sys.exit(f"Missing keys in config: {missing}")

    cfg.setdefault("id_column", "spk_id")
    cfg.setdefault("filters", {})
    cfg.setdefault("weights", {})
    cfg.setdefault("solver", {})
    return cfg


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
    df = apply_filters(df, cfg["filters"])
    df = df.dropna(subset=[cfg["id_column"].lower()]).drop_duplicates(
        subset=[cfg["id_column"].lower()]
    )
    return df.reset_index(drop=True)


def optimize_selection(data, n, targets, weights, id_col, solver_cfg):
    """Pick `n` speakers minimising the weighted deviation from `targets`.

    targets: {attribute: {value: expected_count}}
    weights: {attribute: weight} (default 1)
    Returns the list of selected speaker ids.
    """
    if n > len(data):
        sys.exit(f"Cannot select {n} speakers out of {len(data)} candidates.")

    prob = pulp.LpProblem("speaker_selection", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(data))]
    prob += pulp.lpSum(x) == n  # exactly n speakers

    errors = []
    for col, target_dict in targets.items():
        if col not in data.columns:
            sys.exit(f"Target attribute '{col}' not found in metadata columns.")

        # Pre-compute the row indices of each value (avoids slow per-row lookups)
        rows_by_value = defaultdict(list)
        for i, v in enumerate(data[col].astype(str)):
            rows_by_value[v].append(i)

        weight = weights.get(col, 1)
        for value, target in target_dict.items():
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


def check_targets(name, targets, n):
    for col, d in targets.items():
        total = sum(d.values())
        if total != n:
            print(
                f"WARNING: [{name}] targets for '{col}' sum to {total}, "
                f"expected {n}.",
                file=sys.stderr,
            )


def report(name, df, targets):
    print(f"\n=== {name.upper()} ({len(df)} speakers) ===")
    for col, target_dict in targets.items():
        counts = df[col].astype(str).value_counts()
        print(f"\n{col}:  value  actual / target")
        for value, target in target_dict.items():
            print(f"  {value:>12}  {counts.get(str(value), 0):>4} / {target}")
        extra = set(counts.index) - {str(v) for v in target_dict}
        for value in sorted(extra):
            print(f"  {value:>12}  {counts[value]:>4} / (no target)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    id_col = cfg["id_column"].lower()
    weights = cfg["weights"]
    solver_cfg = cfg["solver"]

    check_targets("enroll", cfg["targets"]["enroll"], cfg["n_enroll"])
    check_targets("trial", cfg["targets"]["trial"], cfg["n_trial"])

    df = load_metadata(cfg)
    print(f"{len(df)} candidate speakers after filtering.")

    print("Optimising enroll set...")
    enroll_ids = optimize_selection(
        df, cfg["n_enroll"], cfg["targets"]["enroll"], weights, id_col, solver_cfg
    )
    enroll_df = df[df[id_col].isin(enroll_ids)].copy()

    print("Optimising trial set (subset of enroll)...")
    trial_ids = optimize_selection(
        enroll_df, cfg["n_trial"], cfg["targets"]["trial"], weights, id_col, solver_cfg
    )
    trial_df = df[df[id_col].isin(trial_ids)].copy()

    enroll_df.to_csv(cfg["output"]["enroll"], index=False)
    trial_df.to_csv(cfg["output"]["trial"], index=False)

    report("enroll", enroll_df, cfg["targets"]["enroll"])
    report("trial", trial_df, cfg["targets"]["trial"])

    print(f"\nSaved: {cfg['output']['enroll']}, {cfg['output']['trial']}")


if __name__ == "__main__":
    main()