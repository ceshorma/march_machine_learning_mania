"""
March Machine Learning Mania 2026 - Fast Baseline
==================================================
End-to-end: data -> Elo + seeds + efficiency -> matchup features -> LightGBM -> submission

Evaluates on past tournaments with temporal CV, then generates both
Stage 1 (2022-2025) and Stage 2 (2026) submissions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
import time

DATA_DIR = Path(__file__).parent.parent / "data"
SUB_DIR = Path(__file__).parent.parent / "submissions"
SUB_DIR.mkdir(exist_ok=True)

CLIP_LOW, CLIP_HIGH = 0.05, 0.95
SEEDS = [42, 2024, 2025, 1234, 5678]

# ── 1. Load Data ─────────────────────────────────────────────────

print("=" * 60)
print("MARCH MACHINE LEARNING MANIA 2026 - BASELINE")
print("=" * 60)

t0 = time.time()
print("\n[1/7] Loading data...")

m_reg = pd.read_csv(DATA_DIR / "MRegularSeasonCompactResults.csv")
m_reg_det = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
m_tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
m_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
m_massey = pd.read_csv(DATA_DIR / "MMasseyOrdinals.csv")
m_conf = pd.read_csv(DATA_DIR / "MTeamConferences.csv")
m_teams = pd.read_csv(DATA_DIR / "MTeams.csv")

w_reg = pd.read_csv(DATA_DIR / "WRegularSeasonCompactResults.csv")
w_reg_det = pd.read_csv(DATA_DIR / "WRegularSeasonDetailedResults.csv")
w_tourney = pd.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")
w_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")
w_conf = pd.read_csv(DATA_DIR / "WTeamConferences.csv")

sub1 = pd.read_csv(DATA_DIR / "SampleSubmissionStage1.csv")
sub2 = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")

print(f"  Men's games: {len(m_reg):,} regular + {len(m_tourney):,} tourney")
print(f"  Women's games: {len(w_reg):,} regular + {len(w_tourney):,} tourney")
print(f"  Stage 1: {len(sub1):,} rows | Stage 2: {len(sub2):,} rows")


# ── 2. Elo Ratings ───────────────────────────────────────────────

print("\n[2/7] Computing Elo ratings...")

def compute_elo(reg_df, conf_df, k=32, home_adv=100, carryover=0.65):
    """Fast Elo computation. Returns {(season, team_id): elo}."""
    ratings = {}
    mean_elo = 1500

    def get(s, t):
        return ratings.get((s, t), mean_elo)

    games = reg_df.sort_values(["Season", "DayNum"])
    seasons = sorted(games["Season"].unique())

    for season in seasons:
        # Season reset
        if season > seasons[0]:
            cm = dict(zip(
                conf_df[conf_df.Season == season]["TeamID"],
                conf_df[conf_df.Season == season]["ConfAbbrev"],
            ))
            if cm:
                conf_elos = {}
                for tid, c in cm.items():
                    conf_elos.setdefault(c, []).append(get(season - 1, tid))
                conf_means = {c: np.mean(v) for c, v in conf_elos.items()}
                for tid, c in cm.items():
                    prev = get(season - 1, tid)
                    ratings[(season, tid)] = carryover * prev + (1 - carryover) * conf_means.get(c, mean_elo)

        sg = games[games.Season == season]
        for _, g in sg.iterrows():
            tw, tl = int(g.WTeamID), int(g.LTeamID)
            ew, el = get(season, tw), get(season, tl)

            # Home adjustment
            loc = g.WLoc
            ew_adj = ew + (home_adv if loc == "H" else 0)
            el_adj = el + (home_adv if loc == "A" else 0)

            expected = 1.0 / (1.0 + 10 ** (-(ew_adj - el_adj) / 400))
            mov = int(g.WScore - g.LScore)
            mov_mult = ((abs(mov) + 3) ** 0.8) / (7.5 + 0.006 * abs(ew - el))

            shift = k * mov_mult * (1 - expected)
            ratings[(season, tw)] = ew + shift
            ratings[(season, tl)] = el - shift

    return ratings

m_elo = compute_elo(m_reg, m_conf)
w_elo = compute_elo(w_reg, w_conf)

# Quick sanity check
latest = max(s for s, _ in m_elo.keys())
top = sorted(
    [(t, e) for (s, t), e in m_elo.items() if s == latest],
    key=lambda x: -x[1]
)[:10]
top_names = {r.TeamID: r.TeamName for _, r in m_teams.iterrows()}
print(f"  Top 10 Men's Elo ({latest}):")
for tid, elo in top:
    print(f"    {top_names.get(tid, tid):25s} {elo:.0f}")


# ── 3. Efficiency from Detailed Results ──────────────────────────

print("\n[3/7] Computing efficiency metrics...")

def compute_efficiency(det_df, max_day=132):
    """Compute season-level efficiency stats per team from detailed box scores."""
    reg = det_df[det_df.DayNum <= max_day].copy()
    rows = []
    for _, g in reg.iterrows():
        s = g.Season
        for prefix, opp_prefix, tid, oid, won, score, opp_score in [
            ("W", "L", g.WTeamID, g.LTeamID, 1, g.WScore, g.LScore),
            ("L", "W", g.LTeamID, g.WTeamID, 0, g.LScore, g.WScore),
        ]:
            fga = g[f"{prefix}FGA"]
            orb = g[f"{prefix}OR"]
            to = g[f"{prefix}TO"]
            fta = g[f"{prefix}FTA"]
            fgm = g[f"{prefix}FGM"]
            fgm3 = g[f"{prefix}FGM3"]
            opp_drb = g[f"{opp_prefix}DR"]

            poss = max(fga - orb + to + 0.475 * fta, 1)
            rows.append({
                "Season": s, "TeamID": int(tid),
                "OffEff": score / poss * 100,
                "DefEff": opp_score / max(g[f"{opp_prefix}FGA"] - g[f"{opp_prefix}OR"] + g[f"{opp_prefix}TO"] + 0.475 * g[f"{opp_prefix}FTA"], 1) * 100,
                "eFG": (fgm + 0.5 * fgm3) / max(fga, 1),
                "TORate": to / poss,
                "ORBRate": orb / max(orb + opp_drb, 1),
                "FTRate": fta / max(fga, 1),
                "Won": won,
            })

    gdf = pd.DataFrame(rows)
    agg = gdf.groupby(["Season", "TeamID"]).agg(
        OffEff=("OffEff", "mean"),
        DefEff=("DefEff", "mean"),
        eFG=("eFG", "mean"),
        TORate=("TORate", "mean"),
        ORBRate=("ORBRate", "mean"),
        FTRate=("FTRate", "mean"),
        WinPct=("Won", "mean"),
        Games=("Won", "count"),
    ).reset_index()
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]
    return agg

m_eff = compute_efficiency(m_reg_det)
w_eff = compute_efficiency(w_reg_det)
print(f"  Men's efficiency: {m_eff.shape}")
print(f"  Women's efficiency: {w_eff.shape}")


# ── 4. Seed + Massey Features ────────────────────────────────────

print("\n[4/7] Building seed & Massey features...")

def parse_seed_num(s):
    return int(s[1:3])

m_seeds["SeedNum"] = m_seeds["Seed"].apply(parse_seed_num)
w_seeds["SeedNum"] = w_seeds["Seed"].apply(parse_seed_num)

# Massey: aggregate top systems (latest day before tourney)
def agg_massey(massey_df, max_day=133):
    df = massey_df[massey_df.RankingDayNum <= max_day]
    latest = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].max().reset_index()
    df = df.merge(latest, on=["Season", "SystemName", "TeamID", "RankingDayNum"])
    agg = df.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).reset_index()
    agg.columns = ["Season", "TeamID", "MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax", "MasseyStd", "MasseyCount"]
    agg["MasseyStd"] = agg["MasseyStd"].fillna(0)
    return agg

m_massey_agg = agg_massey(m_massey)
print(f"  Massey aggregated: {m_massey_agg.shape}")


# ── 5. Build Team Features & Matchup Training Data ──────────────

print("\n[5/7] Building team features & matchup training data...")

def build_team_feats(elo_dict, eff_df, seeds_df, massey_df, conf_df, reg_df, max_day=132):
    """Build per-team-per-season feature vector."""
    # Start with all (season, team) pairs from regular season
    teams_by_season = set()
    for _, g in reg_df[reg_df.DayNum <= max_day].iterrows():
        teams_by_season.add((int(g.Season), int(g.WTeamID)))
        teams_by_season.add((int(g.Season), int(g.LTeamID)))

    rows = []
    for season, tid in teams_by_season:
        row = {"Season": season, "TeamID": tid}
        row["Elo"] = elo_dict.get((season, tid), 1500)

        # Efficiency
        ef = eff_df[(eff_df.Season == season) & (eff_df.TeamID == tid)]
        if len(ef) > 0:
            ef = ef.iloc[0]
            for c in ["OffEff", "DefEff", "NetEff", "eFG", "TORate", "ORBRate", "FTRate", "WinPct"]:
                row[c] = ef[c]

        # Seed
        sd = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == tid)]
        row["SeedNum"] = sd["SeedNum"].values[0] if len(sd) > 0 else 8.5  # default for non-tourney teams

        # Massey
        if massey_df is not None and len(massey_df) > 0:
            ms = massey_df[(massey_df.Season == season) & (massey_df.TeamID == tid)]
            if len(ms) > 0:
                ms = ms.iloc[0]
                for c in ["MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax", "MasseyStd"]:
                    row[c] = ms[c]

        # Win/Loss record
        wins = len(reg_df[(reg_df.Season == season) & (reg_df.WTeamID == tid) & (reg_df.DayNum <= max_day)])
        losses = len(reg_df[(reg_df.Season == season) & (reg_df.LTeamID == tid) & (reg_df.DayNum <= max_day)])
        total = wins + losses
        row["RecordWinPct"] = wins / max(total, 1)
        row["RecordGames"] = total

        rows.append(row)

    return pd.DataFrame(rows)

m_feats = build_team_feats(m_elo, m_eff, m_seeds, m_massey_agg, m_conf, m_reg)
w_feats = build_team_feats(w_elo, w_eff, w_seeds, None, w_conf, w_reg)

print(f"  Men's team features: {m_feats.shape} ({m_feats.columns.tolist()})")
print(f"  Women's team features: {w_feats.shape} ({w_feats.columns.tolist()})")

# Build matchup features (differences)
def make_matchup(team_feats, tourney_df):
    """Create matchup training data from tournament results."""
    feat_cols = [c for c in team_feats.columns if c not in ["Season", "TeamID"]]
    rows = []
    for _, g in tourney_df.iterrows():
        s = int(g.Season)
        t1 = int(min(g.WTeamID, g.LTeamID))
        t2 = int(max(g.WTeamID, g.LTeamID))
        target = 1 if int(g.WTeamID) == t1 else 0

        f1 = team_feats[(team_feats.Season == s) & (team_feats.TeamID == t1)]
        f2 = team_feats[(team_feats.Season == s) & (team_feats.TeamID == t2)]
        if len(f1) == 0 or len(f2) == 0:
            continue
        f1, f2 = f1.iloc[0], f2.iloc[0]

        row = {"Season": s, "TeamID1": t1, "TeamID2": t2, "target": target}
        for c in feat_cols:
            v1 = f1.get(c, np.nan)
            v2 = f2.get(c, np.nan)
            row[f"{c}_diff"] = (v1 - v2) if pd.notna(v1) and pd.notna(v2) else 0
        rows.append(row)

    return pd.DataFrame(rows)

m_train = make_matchup(m_feats, m_tourney)
m_train["is_mens"] = 1
w_train = make_matchup(w_feats, w_tourney)
w_train["is_mens"] = 0

# Align columns
common = sorted(set(m_train.columns) & set(w_train.columns))
train = pd.concat([m_train[common], w_train[common]], ignore_index=True)
train = train.fillna(0)

meta = ["Season", "TeamID1", "TeamID2", "target"]
feat_cols = [c for c in train.columns if c not in meta]

X = train[feat_cols].values
y = train["target"].values
seasons = train["Season"]

print(f"\n  Combined training: {train.shape}")
print(f"  Features ({len(feat_cols)}): {feat_cols}")
print(f"  Target mean: {y.mean():.3f}")
print(f"  Seasons: {int(seasons.min())} - {int(seasons.max())}")


# ── 6. Temporal CV + Seed Baseline ───────────────────────────────

print("\n[6/7] Temporal CV evaluation...")

# Seed-only baseline
seed_col = feat_cols.index("SeedNum_diff") if "SeedNum_diff" in feat_cols else None
if seed_col is not None:
    all_seasons = sorted(seasons.unique())
    seed_scores = []
    for val_season in all_seasons[-10:]:  # last 10 seasons
        tr_mask = seasons < val_season
        va_mask = seasons == val_season
        if va_mask.sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X[tr_mask][:, [seed_col]], y[tr_mask])
        pred = lr.predict_proba(X[va_mask][:, [seed_col]])[:, 1]
        pred = np.clip(pred, CLIP_LOW, CLIP_HIGH)
        score = log_loss(y[va_mask], pred)
        seed_scores.append((int(val_season), score, va_mask.sum()))
        print(f"  Seed baseline | {val_season} | n={va_mask.sum():3d} | LogLoss: {score:.5f}")
    print(f"  Seed baseline MEAN: {np.mean([s[1] for s in seed_scores]):.5f}\n")

# LightGBM temporal CV
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "n_estimators": 300,
    "verbosity": -1,
    "n_jobs": -1,
}

lgb_scores = []
for val_season in all_seasons[-10:]:
    tr_mask = seasons < val_season
    va_mask = seasons == val_season
    if va_mask.sum() == 0:
        continue
    model = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    model.fit(X[tr_mask], y[tr_mask])
    pred = np.clip(model.predict_proba(X[va_mask])[:, 1], CLIP_LOW, CLIP_HIGH)
    score = log_loss(y[va_mask], pred)
    lgb_scores.append((int(val_season), score, va_mask.sum()))
    print(f"  LightGBM      | {val_season} | n={va_mask.sum():3d} | LogLoss: {score:.5f}")

print(f"  LightGBM      MEAN: {np.mean([s[1] for s in lgb_scores]):.5f}")

# Feature importance
model_full = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
model_full.fit(X, y)
imp = sorted(zip(feat_cols, model_full.feature_importances_), key=lambda x: -x[1])
print(f"\n  Top 10 features:")
for name, val in imp[:10]:
    print(f"    {name:30s} {val:5d}")


# ── 7. Generate Submissions ─────────────────────────────────────

print("\n[7/7] Generating submissions...")

def predict_submission(sample_sub, m_feats, w_feats, feat_cols, seeds_list):
    """Generate predictions for a submission file with multi-seed averaging."""
    all_feats = pd.concat([m_feats, w_feats], ignore_index=True)
    fc = [c.replace("_diff", "") for c in feat_cols if c.endswith("_diff")]

    # Build feature matrix for all matchups
    rows = []
    for _, r in sample_sub.iterrows():
        parts = r["ID"].split("_")
        s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])

        f1 = all_feats[(all_feats.Season == s) & (all_feats.TeamID == t1)]
        f2 = all_feats[(all_feats.Season == s) & (all_feats.TeamID == t2)]

        row = {"ID": r["ID"]}
        if len(f1) > 0 and len(f2) > 0:
            f1, f2 = f1.iloc[0], f2.iloc[0]
            for c in fc:
                v1, v2 = f1.get(c, np.nan), f2.get(c, np.nan)
                row[f"{c}_diff"] = (v1 - v2) if pd.notna(v1) and pd.notna(v2) else 0
            # is_mens
            row["is_mens"] = 1 if t1 < 3000 else 0
        else:
            for c in fc:
                row[f"{c}_diff"] = 0
            row["is_mens"] = 1 if t1 < 3000 else 0

        rows.append(row)

    X_sub = pd.DataFrame(rows)
    ids = X_sub["ID"].values
    X_mat = X_sub[feat_cols].fillna(0).values

    # Multi-seed prediction
    preds_all = []
    for seed in seeds_list:
        m = lgb.LGBMClassifier(**{**lgb_params, "random_state": seed})
        m.fit(X, y)
        preds_all.append(m.predict_proba(X_mat)[:, 1])

    avg_pred = np.clip(np.mean(preds_all, axis=0), CLIP_LOW, CLIP_HIGH)

    result = sample_sub.copy()
    result["Pred"] = avg_pred
    return result

# Stage 1
print("  Generating Stage 1 (2022-2025 validation)...")
s1_result = predict_submission(sub1, m_feats, w_feats, feat_cols, SEEDS)
s1_path = SUB_DIR / "submission_stage1_v1.csv"
s1_result.to_csv(s1_path, index=False)
print(f"  Saved: {s1_path}")
print(f"  Stats: mean={s1_result['Pred'].mean():.4f}, std={s1_result['Pred'].std():.4f}, "
      f"min={s1_result['Pred'].min():.4f}, max={s1_result['Pred'].max():.4f}")

# Stage 2
print("\n  Generating Stage 2 (2026 predictions)...")
s2_result = predict_submission(sub2, m_feats, w_feats, feat_cols, SEEDS)
s2_path = SUB_DIR / "submission_stage2_v1.csv"
s2_result.to_csv(s2_path, index=False)
print(f"  Saved: {s2_path}")
print(f"  Stats: mean={s2_result['Pred'].mean():.4f}, std={s2_result['Pred'].std():.4f}, "
      f"min={s2_result['Pred'].min():.4f}, max={s2_result['Pred'].max():.4f}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"DONE in {elapsed:.1f}s")
print(f"{'=' * 60}")
