"""
Experiment B: Blend exp_055 + exp_056 stage2 submissions.
Two best models with different feature sets (31 vs 36 features).
"""
import pandas as pd
from pathlib import Path

SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

sub_055 = pd.read_csv(SUB_DIR / "submission_stage2_exp_055_seedwr_gender_optuna.csv")
sub_056 = pd.read_csv(SUB_DIR / "submission_stage2_exp_056_gender_hc_consist.csv")

assert len(sub_055) == len(sub_056), f"Row count mismatch: {len(sub_055)} vs {len(sub_056)}"
assert (sub_055["ID"] == sub_056["ID"]).all(), "ID mismatch between submissions"

print(f"exp_055: {len(sub_055)} rows, mean={sub_055['Pred'].mean():.5f}, std={sub_055['Pred'].std():.5f}")
print(f"exp_056: {len(sub_056)} rows, mean={sub_056['Pred'].mean():.5f}, std={sub_056['Pred'].std():.5f}")
print()

# Correlation between the two submissions
corr = sub_055["Pred"].corr(sub_056["Pred"])
print(f"Correlation between exp_055 and exp_056 predictions: {corr:.5f}")
print(f"Mean absolute difference: {(sub_055['Pred'] - sub_056['Pred']).abs().mean():.5f}")
print()

for w055, name in [(0.5, "5050"), (0.4, "4060"), (0.3, "3070")]:
    blended = sub_056.copy()
    blended["Pred"] = w055 * sub_055["Pred"] + (1 - w055) * sub_056["Pred"]
    out_path = SUB_DIR / f"submission_stage2_exp_076_blend_{name}.csv"
    blended.to_csv(out_path, index=False)
    print(f"Blend {name} (w055={w055:.1f}): mean={blended['Pred'].mean():.5f}, std={blended['Pred'].std():.5f} -> {out_path.name}")

print("\nDone! Submit the best blend to Kaggle.")
