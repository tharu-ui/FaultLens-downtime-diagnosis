"""Read-only summary of the final fresh-set results. Modifies nothing."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

d = pd.read_csv("results/tables/final_fresh_test_results.csv")
d["answered"] = d["answered"].astype(bool)
d["correct"] = d["correct"].astype(bool)
d["cw"] = d["answered"] & ~d["correct"]

print("Methods:", sorted(d["method"].unique()))
print("Scenarios:", d["scenario_id"].nunique())
print("Defects:", sorted(d["defect"].unique()))
print("Severities:", sorted(d["severity"].unique()))

dfx = d[d["severity"] > 0].copy()

print("\n=== POOLED over defective conditions (severity > 0) ===")
pooled = dfx.groupby("method").agg(
    coverage_pct=("answered", lambda s: round(s.mean() * 100, 1)),
    cw_pct=("cw", lambda s: round(s.mean() * 100, 1)),
)
sel = dfx[dfx["answered"]].groupby("method")["correct"].mean().mul(100).round(1)
pooled["selective_acc_pct"] = sel
print(pooled)

print("\n=== CLEAN (severity 0) ===")
cl = d[d["severity"] == 0].groupby("method").agg(
    coverage_pct=("answered", lambda s: round(s.mean() * 100, 1)),
    cw_pct=("cw", lambda s: round(s.mean() * 100, 1)),
)
print(cl)

print("\n=== CW % by defect x severity ===")
print((d.groupby(["defect", "severity", "method"])["cw"].mean().unstack("method") * 100).round(1))

print("\n=== COVERAGE % by defect x severity ===")
print((d.groupby(["defect", "severity", "method"])["answered"].mean().unstack("method") * 100).round(1))

# Scenario-level bootstrap (resample scenarios, 1000 times)
cwm = dfx.groupby(["scenario_id", "method"])["cw"].mean().unstack("method")
cov = dfx.groupby(["scenario_id", "method"])["answered"].mean().unstack("method")
rng = np.random.default_rng(0)
boots = rng.choice(len(cwm), size=(1000, len(cwm)), replace=True)

def ci(arr):
    m = arr[boots].mean(axis=1)
    return np.round(np.percentile(m, [2.5, 50, 97.5], axis=0) * 100, 1)

print("\n=== POOLED CW % with 95% CI (scenario bootstrap) ===")
q = ci(cwm.to_numpy(dtype=float))
for i, name in enumerate(cwm.columns):
    print(f"{name:28s} {q[1, i]:5.1f}  [{q[0, i]:5.1f}, {q[2, i]:5.1f}]")

if "learned_faultlens" in cwm.columns:
    base = "learned_faultlens"
    print(f"\n=== CW and coverage DIFFERENCE vs {base} (positive = {base} is better) ===")
    for name in cwm.columns:
        if name == base:
            continue
        diff_cw = (cwm[name] - cwm[base]).to_numpy(dtype=float)[:, None]
        diff_cov = (cov[name] - cov[base]).to_numpy(dtype=float)[:, None]
        a = ci(diff_cw)[:, 0]
        b = ci(diff_cov)[:, 0]
        print(f"{name:28s} CW diff {a[1]:5.1f} [{a[0]:5.1f}, {a[2]:5.1f}] pp | coverage diff {b[1]:5.1f} [{b[0]:5.1f}, {b[2]:5.1f}] pp")