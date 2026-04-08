"""
March Machine Learning Mania 2026 - Improved v3
================================================
Improvements over v2:
1. BRIER SCORE evaluation (competition metric since 2023)
2. Elo parameter grid search (K, home_adv, carryover)
3. NaN-native handling (LGB/XGB/CB handle NaN; no fillna(0))
4. Optuna hyperparameter tuning for CatBoost
5. Isotonic calibration on OOF predictions
6. Relaxed clipping [0.02, 0.98] (Brier caps penalty at 1.0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import time
import re
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parent.parent / "data"
EXT_DIR = Path("C:/Users/Admin/Desktop/march_data_temp")
SUB_DIR = Path(__file__).parent.parent / "submissions"
RES_DIR = Path(__file__).parent.parent / "results"
SUB_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Brier score: penalty capped at 1.0, can use wider clip range
CLIP_LOW, CLIP_HIGH = 0.02, 0.98
SEEDS = [42, 2024, 2025, 1234, 5678, 7890, 3141, 2718, 1618, 4242]
N_OPTUNA_TRIALS = 25
EVAL_SEASONS = 10  # Last N seasons for temporal CV

# ── 0. Team Name Mapping ────────────────────────────────────────

EXTERNAL_TO_KAGGLE = {
    "Abilene Christian": "Abilene Chr",
    "Albany": "SUNY Albany",
    "American": "American Univ",
    "Appalachian St.": "Appalachian St",
    "Arizona St.": "Arizona St",
    "Arkansas Little Rock": "Ark Little Rock",
    "Little Rock": "Ark Little Rock",
    "Arkansas Pine Bluff": "Ark Pine Bluff",
    "Arkansas St.": "Arkansas St",
    "Alabama St.": "Alabama St",
    "Ball St.": "Ball St",
    "Boise St.": "Boise St",
    "Boston University": "Boston Univ",
    "Bowling Green St.": "Bowling Green",
    "Cal St. Bakersfield": "CS Bakersfield",
    "Cal St. Fullerton": "CS Fullerton",
    "Cal St. Northridge": "CS Northridge",
    "Cleveland St.": "Cleveland St",
    "Coastal Carolina": "Coastal Car",
    "College of Charleston": "Col Charleston",
    "Colorado St.": "Colorado St",
    "Coppin St.": "Coppin St",
    "East Tennessee St.": "ETSU",
    "Eastern Kentucky": "E Kentucky",
    "Eastern Washington": "E Washington",
    "Fairleigh Dickinson": "F Dickinson",
    "Florida Atlantic": "FL Atlantic",
    "Florida Gulf Coast": "FGCU",
    "Florida St.": "Florida St",
    "Fresno St.": "Fresno St",
    "George Washington": "G Washington",
    "Georgia St.": "Georgia St",
    "Grambling St.": "Grambling",
    "Green Bay": "WI Green Bay",
    "Indiana St.": "Indiana St",
    "Iowa St.": "Iowa St",
    "Jacksonville St.": "Jacksonville St",
    "Kansas St.": "Kansas St",
    "Kennesaw St.": "Kennesaw",
    "Kent St.": "Kent",
    "Long Beach St.": "Long Beach St",
    "Louisiana Lafayette": "Louisiana",
    "Loyola Chicago": "Loyola-Chicago",
    "McNeese St.": "McNeese St",
    "Michigan St.": "Michigan St",
    "Middle Tennessee": "MTSU",
    "Milwaukee": "WI Milwaukee",
    "Mississippi St.": "Mississippi St",
    "Mississippi Valley St.": "MS Valley St",
    "Montana St.": "Montana St",
    "Morehead St.": "Morehead St",
    "Morgan St.": "Morgan St",
    "Mount St. Mary's": "Mt St Mary's",
    "Murray St.": "Murray St",
    "Nebraska Omaha": "NE Omaha",
    "New Mexico St.": "New Mexico St",
    "Norfolk St.": "Norfolk St",
    "North Carolina A&T": "NC A&T",
    "North Carolina Central": "NC Central",
    "North Carolina St.": "NC State",
    "North Dakota St.": "N Dakota St",
    "Northern Colorado": "N Colorado",
    "Northern Kentucky": "N Kentucky",
    "Northwestern St.": "Northwestern LA",
    "Ohio St.": "Ohio St",
    "Oklahoma St.": "Oklahoma St",
    "Oregon St.": "Oregon St",
    "Penn St.": "Penn St",
    "Portland St.": "Portland St",
    "Prairie View A&M": "Prairie View",
    "SIU Edwardsville": "SIUE",
    "Saint Francis": "St Francis PA",
    "Saint Joseph's": "St Joseph's PA",
    "Saint Louis": "St Louis",
    "Saint Mary's": "St Mary's CA",
    "Saint Peter's": "St Peter's",
    "Sam Houston St.": "Sam Houston St",
    "San Diego St.": "San Diego St",
    "South Dakota St.": "S Dakota St",
    "Southeast Missouri St.": "SE Missouri St",
    "Southern": "Southern Univ",
    "St. Bonaventure": "St Bonaventure",
    "St. John's": "St John's",
    "Stephen F. Austin": "SF Austin",
    "Texas A&M Corpus Chris": "TAM C. Christi",
    "Texas Southern": "TX Southern",
    "UTSA": "UT San Antonio",
    "Utah St.": "Utah St",
    "Washington St.": "Washington St",
    "Weber St.": "Weber St",
    "Western Kentucky": "WKU",
    "Western Michigan": "W Michigan",
    "Wichita St.": "Wichita St",
    "Wright St.": "Wright St",
    "Illinois St.": "Illinois St",
    "Missouri St.": "Missouri St",
    "Nicholls St.": "Nicholls St",
    "South Carolina St.": "S Carolina St",
    "Delaware St.": "Delaware St",
    "Idaho St.": "Idaho St",
    "Jackson St.": "Jackson St",
    "Tennessee St.": "Tennessee St",
    "Youngstown St.": "Youngstown St",
    "Chicago St.": "Chicago St",
    "Tarleton St.": "Tarleton St",
    "Sacramento St.": "CS Sacramento",
    "Southern Illinois": "S Illinois",
    "Texas St.": "Texas St",
    "Eastern Michigan": "E Michigan",
    "Eastern Illinois": "E Illinois",
    "Central Michigan": "C Michigan",
    "Northern Illinois": "N Illinois",
    "Western Illinois": "W Illinois",
    "Southeast Louisiana": "SE Louisiana",
    "Loyola Marymount": "Loy Marymount",
    "San Jose St.": "San Jose St",
    "Southern Utah": "Southern Utah",
    "Central Arkansas": "Cent Arkansas",
    "Houston Christian": "Houston Chr",
    "Central Connecticut": "Central Conn",
    # Additional mappings to fix unmapped teams
    "Bethune Cookman": "Bethune-Cookman",
    "Bethune-Cookman": "Bethune-Cookman",
    "Charleston": "Col Charleston",
    "Detroit Mercy": "Detroit",
    "Detroit": "Detroit",
    "FIU": "FL International",
    "Georgia Southern": "Ga Southern",
    "Houston Baptist": "Houston Chr",
    "LIU Brooklyn": "LIU",
    "LIU": "LIU",
    "Loyola Maryland": "Loyola MD",
    "Loyola (MD)": "Loyola MD",
    "Massachusetts": "Massachusetts",
    "UMass": "Massachusetts",
    "UMass Lowell": "MA Lowell",
    "Maryland Eastern Shore": "MD E Shore",
    "Miami (FL)": "Miami FL",
    "Miami FL": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Miami OH": "Miami OH",
    "North Carolina": "North Carolina",
    "North Carolina Asheville": "NC Asheville",
    "North Carolina Greensboro": "NC Greensboro",
    "North Carolina Wilmington": "NC Wilmington",
    "Penn": "Pennsylvania",
    "Pennsylvania": "Pennsylvania",
    "Purdue Fort Wayne": "PFW",
    "IPFW": "PFW",
    "South Carolina Upstate": "USC Upstate",
    "Southern Miss": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
    "St. Francis (PA)": "St Francis PA",
    "St. Francis Brooklyn": "St Francis NY",
    "St. Joseph's": "St Joseph's PA",
    "St. Mary's (CA)": "St Mary's CA",
    "St. Peter's": "St Peter's",
    "Tennessee Martin": "UT Martin",
    "UT Martin": "UT Martin",
    "Texas A&M Corpus Christi": "TAM C. Christi",
    "Texas Arlington": "UT Arlington",
    "UT Arlington": "UT Arlington",
    "Texas Rio Grande Valley": "UT Rio Grande",
    "UTRGV": "UT Rio Grande",
    "UC Davis": "UC Davis",
    "UC Irvine": "UC Irvine",
    "UC Riverside": "UC Riverside",
    "UC Santa Barbara": "UC Santa Barbara",
    "VCU": "VA Commonwealth",
    "Virginia Commonwealth": "VA Commonwealth",
    "Virginia Military Institute": "VMI",
    "VMI": "VMI",
    "Omaha": "NE Omaha",
}


def normalize_name(name):
    name = str(name).strip()
    if name in EXTERNAL_TO_KAGGLE:
        return EXTERNAL_TO_KAGGLE[name]
    name = re.sub(r'\.(?=\s|$)', '', name)
    return name


# ── 1. Load Data ─────────────────────────────────────────────────

print("=" * 60)
print("MARCH MACHINE LEARNING MANIA 2026 - IMPROVED v3")
print("=" * 60)

t0 = time.time()
print("\n[1/10] Loading data...")

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

print(f"  Men: {len(m_reg):,} reg + {len(m_tourney):,} tourney")
print(f"  Women: {len(w_reg):,} reg + {len(w_tourney):,} tourney")

# ── 2. Load & Map External Data ─────────────────────────────────

print("\n[2/10] Loading external data (KenPom + Barttorvik)...")

name_to_id = dict(zip(m_teams["TeamName"], m_teams["TeamID"]))

# KenPom
kenpom_eff = pd.read_csv(EXT_DIR / "kenpom/INT _ KenPom _ Efficiency.csv")
kenpom_eff["KaggleName"] = kenpom_eff["Team"].apply(normalize_name)
kenpom_eff["TeamID"] = kenpom_eff["KaggleName"].map(name_to_id)

# Barttorvik
bart_files = []
for yr in range(13, 26):
    f = EXT_DIR / f"barttorvik/cbb{yr}.csv"
    if f.exists():
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        if "YEAR" not in df.columns:
            df["YEAR"] = 2000 + yr
        bart_files.append(df)

bart_combined = pd.read_csv(EXT_DIR / "barttorvik/cbb.csv")
bart_combined.columns = [c.strip().upper() for c in bart_combined.columns]
bart_files.append(bart_combined)

bart = pd.concat(bart_files, ignore_index=True).drop_duplicates(subset=["TEAM", "YEAR"], keep="last")
bart["KaggleName"] = bart["TEAM"].apply(normalize_name)
bart["TeamID"] = bart["KaggleName"].map(name_to_id)

kp_mapped = kenpom_eff["TeamID"].notna().sum()
bart_mapped = bart["TeamID"].notna().sum()
print(f"  KenPom: {kp_mapped}/{len(kenpom_eff)} mapped ({kp_mapped/len(kenpom_eff)*100:.1f}%)")
print(f"  Barttorvik: {bart_mapped}/{len(bart)} mapped ({bart_mapped/len(bart)*100:.1f}%)")

kp_unmapped = kenpom_eff[kenpom_eff["TeamID"].isna()]["Team"].unique()
bart_unmapped = bart[bart["TeamID"].isna()]["TEAM"].unique()
if len(kp_unmapped) > 0:
    print(f"  KenPom unmapped ({len(kp_unmapped)}): {sorted(kp_unmapped)[:10]}...")
if len(bart_unmapped) > 0:
    print(f"  Barttorvik unmapped ({len(bart_unmapped)}): {sorted(bart_unmapped)[:10]}...")

# Build clean merge tables
kp_clean = kenpom_eff[["Season", "TeamID"]].copy()
kp_eff_cols = kenpom_eff.columns.tolist()
for c in kp_eff_cols:
    cl = c.lower().strip()
    if "adjusted offensive" in cl:
        kp_clean["AdjOE"] = kenpom_eff[c]
    elif "adjusted defensive" in cl:
        kp_clean["AdjDE"] = kenpom_eff[c]
    elif "adjusted tem" in cl and "rank" not in cl:
        kp_clean["AdjTempo"] = kenpom_eff[c]
if "AdjOE" in kp_clean.columns and "AdjDE" in kp_clean.columns:
    kp_clean["AdjEM"] = kp_clean["AdjOE"] - kp_clean["AdjDE"]
kp_clean = kp_clean.dropna(subset=["TeamID"]).copy()
kp_clean["TeamID"] = kp_clean["TeamID"].astype(int)

bart_clean = bart[["YEAR", "TeamID"]].copy()
for src, dst in [("BARTHAG", "BARTHAG"), ("ADJOE", "ADJOE"), ("ADJDE", "ADJDE"),
                 ("EFG_O", "EFG_O"), ("EFG_D", "EFG_D"), ("WAB", "WAB"), ("ADJ_T", "ADJ_T")]:
    if src in bart.columns:
        bart_clean[dst] = bart[src].values
bart_clean = bart_clean.dropna(subset=["TeamID"]).copy()
bart_clean["TeamID"] = bart_clean["TeamID"].astype(int)

print(f"  KenPom clean: {kp_clean.shape}, Barttorvik clean: {bart_clean.shape}")


# ── 3. Elo Grid Search ──────────────────────────────────────────

print("\n[3/10] Elo parameter grid search...")


def compute_elo(reg_df, conf_df, k=32, home_adv=100, carryover=0.65):
    ratings = {}
    mean_elo = 1500

    def get(s, t):
        return ratings.get((s, t), mean_elo)

    games = reg_df.sort_values(["Season", "DayNum"])
    seasons = sorted(games["Season"].unique())

    for season in seasons:
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


def quick_elo_eval(elo_dict, seeds_df, tourney_df, eval_seasons=5):
    """Fast evaluation: LR on Elo_diff + SeedNum_diff, Brier score."""
    seeds_map = {}
    for _, r in seeds_df.iterrows():
        seeds_map[(r.Season, r.TeamID)] = int(r.Seed[1:3])

    rows = []
    for _, g in tourney_df.iterrows():
        s = int(g.Season)
        t1, t2 = int(min(g.WTeamID, g.LTeamID)), int(max(g.WTeamID, g.LTeamID))
        target = 1 if int(g.WTeamID) == t1 else 0
        e1, e2 = elo_dict.get((s, t1), 1500), elo_dict.get((s, t2), 1500)
        s1, s2 = seeds_map.get((s, t1), 8.5), seeds_map.get((s, t2), 8.5)
        rows.append({"Season": s, "elo_diff": e1 - e2, "seed_diff": s1 - s2, "target": target})

    df = pd.DataFrame(rows)
    all_seasons = sorted(df["Season"].unique())
    eval_szns = all_seasons[-eval_seasons:]

    scores = []
    for vs in eval_szns:
        tr = df[df.Season < vs]
        va = df[df.Season == vs]
        if len(va) == 0:
            continue
        X_tr = tr[["elo_diff", "seed_diff"]].values
        X_va = va[["elo_diff", "seed_diff"]].values
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr, tr["target"].values)
        pred = np.clip(lr.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
        scores.append(brier_score_loss(va["target"].values, pred))
    return np.mean(scores) if scores else 1.0


# Grid search on Men's data (more data, more signal)
elo_grid = {
    "k": [24, 32, 40, 48],
    "home_adv": [64, 82, 100],
    "carryover": [0.55, 0.65, 0.75],
}

best_brier = 1.0
best_elo_params = {"k": 32, "home_adv": 100, "carryover": 0.65}
grid_results = []

m_seeds_parsed = m_seeds.copy()
m_seeds_parsed["SeedNum"] = m_seeds_parsed["Seed"].apply(lambda s: int(s[1:3]))

total_combos = len(elo_grid["k"]) * len(elo_grid["home_adv"]) * len(elo_grid["carryover"])
combo_i = 0

for k in elo_grid["k"]:
    for ha in elo_grid["home_adv"]:
        for co in elo_grid["carryover"]:
            combo_i += 1
            elo = compute_elo(m_reg, m_conf, k=k, home_adv=ha, carryover=co)
            score = quick_elo_eval(elo, m_seeds_parsed, m_tourney)
            grid_results.append({"k": k, "home_adv": ha, "carryover": co, "brier": score})
            if score < best_brier:
                best_brier = score
                best_elo_params = {"k": k, "home_adv": ha, "carryover": co}
            if combo_i % 12 == 0 or combo_i == total_combos:
                print(f"  Grid search: {combo_i}/{total_combos} combos tested...")

print(f"  Best Elo params: K={best_elo_params['k']}, home_adv={best_elo_params['home_adv']}, carryover={best_elo_params['carryover']}")
print(f"  Best Brier (quick eval): {best_brier:.5f}")

# Show top 5 combos
grid_results.sort(key=lambda x: x["brier"])
print("  Top 5 Elo configs:")
for r in grid_results[:5]:
    print(f"    K={r['k']:2d} | home={r['home_adv']:3d} | carry={r['carryover']:.2f} | Brier={r['brier']:.5f}")

# Compute Elo with best params
m_elo = compute_elo(m_reg, m_conf, **best_elo_params)
w_elo = compute_elo(w_reg, w_conf, **best_elo_params)

top_names = {r.TeamID: r.TeamName for _, r in m_teams.iterrows()}
latest = max(s for s, _ in m_elo.keys())
top = sorted([(t, e) for (s, t), e in m_elo.items() if s == latest], key=lambda x: -x[1])[:10]
print(f"  Top 10 Men's Elo ({latest}):")
for tid, elo in top:
    print(f"    {top_names.get(tid, tid):25s} {elo:.0f}")


# ── 4. Efficiency from Detailed Results ─────────────────────────

print("\n[4/10] Computing efficiency metrics...")


def compute_efficiency(det_df, max_day=132):
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
            opp_poss = max(g[f"{opp_prefix}FGA"] - g[f"{opp_prefix}OR"] + g[f"{opp_prefix}TO"] + 0.475 * g[f"{opp_prefix}FTA"], 1)
            rows.append({
                "Season": s, "TeamID": int(tid),
                "OffEff": score / poss * 100,
                "DefEff": opp_score / opp_poss * 100,
                "eFG": (fgm + 0.5 * fgm3) / max(fga, 1),
                "TORate": to / poss,
                "ORBRate": orb / max(orb + opp_drb, 1),
                "FTRate": fta / max(fga, 1),
                "Won": won,
            })

    gdf = pd.DataFrame(rows)
    agg = gdf.groupby(["Season", "TeamID"]).agg(
        OffEff=("OffEff", "mean"), DefEff=("DefEff", "mean"),
        eFG=("eFG", "mean"), TORate=("TORate", "mean"),
        ORBRate=("ORBRate", "mean"), FTRate=("FTRate", "mean"),
        WinPct=("Won", "mean"), Games=("Won", "count"),
    ).reset_index()
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]
    return agg


m_eff = compute_efficiency(m_reg_det)
w_eff = compute_efficiency(w_reg_det)
print(f"  Men's: {m_eff.shape}, Women's: {w_eff.shape}")


# ── 5. Seed + Massey Features ──────────────────────────────────

print("\n[5/10] Building seed & Massey features...")

m_seeds["SeedNum"] = m_seeds["Seed"].apply(lambda s: int(s[1:3]))
w_seeds["SeedNum"] = w_seeds["Seed"].apply(lambda s: int(s[1:3]))


def agg_massey(massey_df, max_day=133):
    df = massey_df[massey_df.RankingDayNum <= max_day]
    latest = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].max().reset_index()
    df = df.merge(latest, on=["Season", "SystemName", "TeamID", "RankingDayNum"])
    agg = df.groupby(["Season", "TeamID"])["OrdinalRank"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).reset_index()
    agg.columns = ["Season", "TeamID", "MasseyMean", "MasseyMedian",
                    "MasseyMin", "MasseyMax", "MasseyStd", "MasseyCount"]
    agg["MasseyStd"] = agg["MasseyStd"].fillna(0)
    return agg


m_massey_agg = agg_massey(m_massey)
print(f"  Massey Men's: {m_massey_agg.shape}")
# No Women's Massey Ordinals in Kaggle data


# ── 6. Build Team Features ──────────────────────────────────────

print("\n[6/10] Building team features with external data...")


def build_team_feats(elo_dict, eff_df, seeds_df, massey_df, conf_df, reg_df,
                     kenpom_df=None, bart_df=None, max_day=132):
    teams_by_season = set()
    for _, g in reg_df[reg_df.DayNum <= max_day].iterrows():
        teams_by_season.add((int(g.Season), int(g.WTeamID)))
        teams_by_season.add((int(g.Season), int(g.LTeamID)))

    rows = []
    for season, tid in teams_by_season:
        row = {"Season": season, "TeamID": tid}
        row["Elo"] = elo_dict.get((season, tid), 1500)

        ef = eff_df[(eff_df.Season == season) & (eff_df.TeamID == tid)]
        if len(ef) > 0:
            ef = ef.iloc[0]
            for c in ["OffEff", "DefEff", "NetEff", "eFG", "TORate", "ORBRate", "FTRate", "WinPct"]:
                row[c] = ef[c]

        sd = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == tid)]
        row["SeedNum"] = sd["SeedNum"].values[0] if len(sd) > 0 else np.nan  # NaN for non-tournament teams

        if massey_df is not None and len(massey_df) > 0:
            ms = massey_df[(massey_df.Season == season) & (massey_df.TeamID == tid)]
            if len(ms) > 0:
                ms = ms.iloc[0]
                for c in ["MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax", "MasseyStd"]:
                    row[c] = ms[c]

        if kenpom_df is not None:
            kp = kenpom_df[(kenpom_df.Season == season) & (kenpom_df.TeamID == tid)]
            if len(kp) > 0:
                kp = kp.iloc[0]
                for col in ["AdjOE", "AdjDE", "AdjEM", "AdjTempo"]:
                    if col in kp.index:
                        row[f"KP_{col}"] = kp[col]

        if bart_df is not None:
            bt = bart_df[(bart_df.YEAR == season) & (bart_df.TeamID == tid)]
            if len(bt) > 0:
                bt = bt.iloc[0]
                for src_col, dst_col in [
                    ("BARTHAG", "BT_BARTHAG"), ("ADJOE", "BT_AdjOE"),
                    ("ADJDE", "BT_AdjDE"), ("EFG_O", "BT_eFG_O"),
                    ("EFG_D", "BT_eFG_D"), ("WAB", "BT_WAB"),
                    ("ADJ_T", "BT_AdjTempo"),
                ]:
                    if src_col in bt.index:
                        row[dst_col] = bt[src_col]

        # Win/Loss record
        wins = len(reg_df[(reg_df.Season == season) & (reg_df.WTeamID == tid) & (reg_df.DayNum <= max_day)])
        losses = len(reg_df[(reg_df.Season == season) & (reg_df.LTeamID == tid) & (reg_df.DayNum <= max_day)])
        total = wins + losses
        row["RecordWinPct"] = wins / max(total, 1)

        rows.append(row)

    return pd.DataFrame(rows)


m_feats = build_team_feats(
    m_elo, m_eff, m_seeds, m_massey_agg, m_conf, m_reg,
    kenpom_df=kp_clean, bart_df=bart_clean
)
w_feats = build_team_feats(
    w_elo, w_eff, w_seeds, None, w_conf, w_reg,
    kenpom_df=None, bart_df=None
)

print(f"  Men's: {m_feats.shape}, Women's: {w_feats.shape}")


# ── 7. Build Matchup Training Data ─────────────────────────────

print("\n[7/10] Building matchup training data...")


def make_matchup(team_feats, tourney_df, feat_cols_list):
    rows = []
    for _, g in tourney_df.iterrows():
        s = int(g.Season)
        t1, t2 = int(min(g.WTeamID, g.LTeamID)), int(max(g.WTeamID, g.LTeamID))
        target = 1 if int(g.WTeamID) == t1 else 0

        f1 = team_feats[(team_feats.Season == s) & (team_feats.TeamID == t1)]
        f2 = team_feats[(team_feats.Season == s) & (team_feats.TeamID == t2)]
        if len(f1) == 0 or len(f2) == 0:
            continue
        f1, f2 = f1.iloc[0], f2.iloc[0]

        row = {"Season": s, "TeamID1": t1, "TeamID2": t2, "target": target}
        for c in feat_cols_list:
            v1 = f1.get(c, np.nan)
            v2 = f2.get(c, np.nan)
            # Keep NaN if either value is missing (models handle NaN natively)
            if pd.notna(v1) and pd.notna(v2):
                row[f"{c}_diff"] = v1 - v2
            else:
                row[f"{c}_diff"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


exclude = {"Season", "TeamID", "Games", "RecordGames"}
m_feat_cols = [c for c in m_feats.columns if c not in exclude and m_feats[c].dtype in [np.float64, np.int64, np.float32]]
w_feat_cols = [c for c in w_feats.columns if c not in exclude and w_feats[c].dtype in [np.float64, np.int64, np.float32]]

m_train = make_matchup(m_feats, m_tourney, m_feat_cols)
m_train["is_mens"] = 1
w_train = make_matchup(w_feats, w_tourney, w_feat_cols)
w_train["is_mens"] = 0

# Align columns - keep NaN (no fillna!)
all_cols = sorted(set(m_train.columns) | set(w_train.columns))
for c in all_cols:
    if c not in m_train.columns:
        m_train[c] = np.nan
    if c not in w_train.columns:
        w_train[c] = np.nan

train = pd.concat([m_train[all_cols], w_train[all_cols]], ignore_index=True)
# NO fillna(0)! LGB, XGB, CB all handle NaN natively.

meta = ["Season", "TeamID1", "TeamID2", "target"]
feat_cols = sorted([c for c in train.columns if c not in meta])

X = train[feat_cols].values.astype(np.float32)
y = train["target"].values
seasons = train["Season"].values

print(f"  Combined training: {train.shape}")
print(f"  Features ({len(feat_cols)}): {feat_cols}")
print(f"  NaN distribution:")
nan_counts = train[feat_cols].isna().sum()
nan_feats = nan_counts[nan_counts > 0]
if len(nan_feats) > 0:
    for fname, cnt in nan_feats.items():
        print(f"    {fname}: {cnt} NaN ({cnt/len(train)*100:.1f}%)")
else:
    print(f"    No NaN values")


# ── 8. Optuna Tuning for CatBoost ──────────────────────────────

print(f"\n[8/10] Optuna CatBoost tuning ({N_OPTUNA_TRIALS} trials)...")

all_seasons = sorted(np.unique(seasons))
eval_szns = all_seasons[-EVAL_SEASONS:]


def objective(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "iterations": trial.suggest_int("iterations", 200, 800),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "verbose": 0,
    }

    scores = []
    for vs in eval_szns:
        tr_idx = seasons < vs
        va_idx = seasons == vs
        if va_idx.sum() == 0:
            continue
        model = CatBoostClassifier(**{**params, "random_seed": 42})
        model.fit(X[tr_idx], y[tr_idx])
        pred = np.clip(model.predict_proba(X[va_idx])[:, 1], CLIP_LOW, CLIP_HIGH)
        scores.append(brier_score_loss(y[va_idx], pred))

    return np.mean(scores)


study = optuna.create_study(direction="minimize", study_name="catboost_v3")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

best_cb_params = study.best_params
best_cb_params["loss_function"] = "Logloss"
best_cb_params["eval_metric"] = "Logloss"
best_cb_params["verbose"] = 0

print(f"  Best Brier: {study.best_value:.5f}")
print(f"  Best params: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in best_cb_params.items()}, indent=4)}")


# ── 9. Temporal CV with Brier Score ─────────────────────────────

print(f"\n[9/10] Temporal CV (Brier Score, last {EVAL_SEASONS} seasons)...")

lgb_params = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "n_estimators": 500,
    "verbosity": -1, "n_jobs": -1,
}

xgb_params = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "n_estimators": 500,
    "tree_method": "hist", "verbosity": 0, "n_jobs": -1,
}

# OOF predictions
lgb_oof = np.full(len(y), np.nan)
xgb_oof = np.full(len(y), np.nan)
cb_oof = np.full(len(y), np.nan)

lgb_scores, xgb_scores, cb_scores = [], [], []

# Seed baseline with Brier
seed_col = feat_cols.index("SeedNum_diff") if "SeedNum_diff" in feat_cols else None
if seed_col is not None:
    seed_scores_brier = []
    for vs in eval_szns:
        tr_mask = seasons < vs
        va_mask = seasons == vs
        if va_mask.sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000)
        # Seed baseline: use only non-NaN seed values
        X_seed_tr = X[tr_mask][:, [seed_col]]
        X_seed_va = X[va_mask][:, [seed_col]]
        # Fill NaN seeds with 0 for LR (LR can't handle NaN)
        X_seed_tr = np.nan_to_num(X_seed_tr, nan=0.0)
        X_seed_va = np.nan_to_num(X_seed_va, nan=0.0)
        lr.fit(X_seed_tr, y[tr_mask])
        pred = np.clip(lr.predict_proba(X_seed_va)[:, 1], CLIP_LOW, CLIP_HIGH)
        seed_scores_brier.append(brier_score_loss(y[va_mask], pred))
    print(f"  Seed baseline Brier: {np.mean(seed_scores_brier):.5f}")

print("\n  Per-season results:")
for vs in eval_szns:
    tr_mask = seasons < vs
    va_mask = seasons == vs
    if va_mask.sum() == 0:
        continue

    X_tr, X_va = X[tr_mask], X[va_mask]
    y_tr, y_va = y[tr_mask], y[va_mask]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    m_lgb.fit(X_tr, y_tr)
    p_lgb = np.clip(m_lgb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    lgb_oof[va_mask] = p_lgb
    s_lgb = brier_score_loss(y_va, p_lgb)
    lgb_scores.append(s_lgb)

    # XGBoost
    m_xgb = xgb.XGBClassifier(**{**xgb_params, "random_state": 42})
    m_xgb.fit(X_tr, y_tr)
    p_xgb = np.clip(m_xgb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    xgb_oof[va_mask] = p_xgb
    s_xgb = brier_score_loss(y_va, p_xgb)
    xgb_scores.append(s_xgb)

    # CatBoost (Optuna params)
    m_cb = CatBoostClassifier(**{**best_cb_params, "random_seed": 42})
    m_cb.fit(X_tr, y_tr)
    p_cb = np.clip(m_cb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    cb_oof[va_mask] = p_cb
    s_cb = brier_score_loss(y_va, p_cb)
    cb_scores.append(s_cb)

    print(f"  {int(vs)} | n={int(va_mask.sum()):3d} | LGB: {s_lgb:.5f} | XGB: {s_xgb:.5f} | CB: {s_cb:.5f}")

print(f"\n  LightGBM  MEAN Brier: {np.mean(lgb_scores):.5f}")
print(f"  XGBoost   MEAN Brier: {np.mean(xgb_scores):.5f}")
print(f"  CatBoost  MEAN Brier: {np.mean(cb_scores):.5f}")

# Ensemble: simple avg + hill climbing with BRIER SCORE
valid = ~(np.isnan(lgb_oof) | np.isnan(xgb_oof) | np.isnan(cb_oof))
avg_oof = (lgb_oof[valid] + xgb_oof[valid] + cb_oof[valid]) / 3
avg_brier = brier_score_loss(y[valid], np.clip(avg_oof, CLIP_LOW, CLIP_HIGH))
print(f"  Ensemble AVG  Brier: {avg_brier:.5f}")


def hill_climbing(preds_list, y_true, n_iter=500, step=0.01):
    """Hill climbing with Brier score."""
    n = len(preds_list)
    weights = [1 / n] * n
    best_score = brier_score_loss(y_true, np.clip(sum(w * p for w, p in zip(weights, preds_list)), CLIP_LOW, CLIP_HIGH))

    for _ in range(n_iter):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_w = weights.copy()
                new_w[i] = max(0, new_w[i] + step)
                new_w[j] = max(0, new_w[j] - step)
                s = sum(new_w)
                if s == 0:
                    continue
                new_w = [w / s for w in new_w]
                pred = np.clip(sum(w * p for w, p in zip(new_w, preds_list)), CLIP_LOW, CLIP_HIGH)
                score = brier_score_loss(y_true, pred)
                if score < best_score:
                    weights = new_w
                    best_score = score
                    improved = True
        if not improved:
            break

    return weights, best_score


hc_weights, hc_brier = hill_climbing(
    [lgb_oof[valid], xgb_oof[valid], cb_oof[valid]], y[valid]
)
print(f"  Ensemble HC   Brier: {hc_brier:.5f}")
print(f"  Weights: LGB={hc_weights[0]:.3f}, XGB={hc_weights[1]:.3f}, CB={hc_weights[2]:.3f}")

# Also report log loss for reference
avg_ll = log_loss(y[valid], np.clip(avg_oof, CLIP_LOW, CLIP_HIGH))
hc_ll = log_loss(y[valid], np.clip(
    sum(w * p for w, p in zip(hc_weights, [lgb_oof[valid], xgb_oof[valid], cb_oof[valid]])),
    CLIP_LOW, CLIP_HIGH
))
print(f"\n  (Reference) Ensemble AVG LogLoss: {avg_ll:.5f}")
print(f"  (Reference) Ensemble HC  LogLoss: {hc_ll:.5f}")


# ── 9b. Isotonic Calibration ───────────────────────────────────

print("\n  Isotonic calibration on OOF...")

# Ensemble OOF predictions (using HC weights)
ens_oof = np.full(len(y), np.nan)
ens_oof[valid] = sum(w * p for w, p in zip(hc_weights, [lgb_oof[valid], xgb_oof[valid], cb_oof[valid]]))

# Fit isotonic regression on OOF using temporal splits
iso_scores = []
iso_oof = np.full(len(y), np.nan)

for vs in eval_szns:
    va_mask = seasons == vs
    tr_mask = (seasons < vs) & valid
    if va_mask.sum() == 0 or tr_mask.sum() < 20:
        continue

    iso = IsotonicRegression(y_min=CLIP_LOW, y_max=CLIP_HIGH, out_of_bounds="clip")
    iso.fit(ens_oof[tr_mask], y[tr_mask])

    if va_mask.sum() > 0 and valid[va_mask].all():
        cal_pred = iso.predict(ens_oof[va_mask])
        cal_pred = np.clip(cal_pred, CLIP_LOW, CLIP_HIGH)
        iso_oof[va_mask] = cal_pred
        iso_score = brier_score_loss(y[va_mask], cal_pred)
        iso_scores.append(iso_score)

if iso_scores:
    print(f"  Isotonic calibrated Brier: {np.mean(iso_scores):.5f}")
    iso_valid = ~np.isnan(iso_oof)
    iso_ll = log_loss(y[iso_valid], np.clip(iso_oof[iso_valid], CLIP_LOW, CLIP_HIGH))
    print(f"  (Reference) Isotonic LogLoss: {iso_ll:.5f}")

    # Decide: use calibration only if it improves
    use_isotonic = np.mean(iso_scores) < hc_brier
    print(f"  Use isotonic: {use_isotonic} (iso={np.mean(iso_scores):.5f} vs raw={hc_brier:.5f})")
else:
    use_isotonic = False
    print(f"  Isotonic calibration: not enough data")

# Feature importance
model_full = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
model_full.fit(X, y)
imp = sorted(zip(feat_cols, model_full.feature_importances_), key=lambda x: -x[1])
print(f"\n  Top 15 features:")
for name, val in imp[:15]:
    print(f"    {name:35s} {val:5d}")


# ── 10. Generate Submissions ────────────────────────────────────

print("\n[10/10] Generating submissions with calibrated 3-model ensemble...")

# Train final isotonic calibrator on ALL OOF data if using
final_iso = None
if use_isotonic:
    final_iso = IsotonicRegression(y_min=CLIP_LOW, y_max=CLIP_HIGH, out_of_bounds="clip")
    final_iso.fit(ens_oof[valid], y[valid])
    print("  Fitted final isotonic calibrator on all OOF data")


def predict_submission(sample_sub, m_feats, w_feats, feat_cols, seeds_list, weights, iso_model=None):
    all_feats = pd.concat([m_feats, w_feats], ignore_index=True)
    fc = set()
    for c in feat_cols:
        if c.endswith("_diff"):
            fc.add(c.replace("_diff", ""))
        elif c == "is_mens":
            pass

    rows = []
    for _, r in sample_sub.iterrows():
        parts = r["ID"].split("_")
        s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])

        f1 = all_feats[(all_feats.Season == s) & (all_feats.TeamID == t1)]
        f2 = all_feats[(all_feats.Season == s) & (all_feats.TeamID == t2)]

        row = {"ID": r["ID"]}
        if len(f1) > 0 and len(f2) > 0:
            f1i, f2i = f1.iloc[0], f2.iloc[0]
            for c in fc:
                v1, v2 = f1i.get(c, np.nan), f2i.get(c, np.nan)
                if pd.notna(v1) and pd.notna(v2):
                    row[f"{c}_diff"] = v1 - v2
                else:
                    row[f"{c}_diff"] = np.nan
            row["is_mens"] = 1 if t1 < 3000 else 0
        else:
            for c in fc:
                row[f"{c}_diff"] = np.nan
            row["is_mens"] = 1 if t1 < 3000 else 0
        rows.append(row)

    X_sub = pd.DataFrame(rows)
    ids = X_sub["ID"].values

    for c in feat_cols:
        if c not in X_sub.columns:
            X_sub[c] = np.nan
    X_mat = X_sub[feat_cols].values.astype(np.float32)

    # Multi-seed 3-model prediction
    lgb_preds_all, xgb_preds_all, cb_preds_all = [], [], []

    for seed in seeds_list:
        m_l = lgb.LGBMClassifier(**{**lgb_params, "random_state": seed})
        m_l.fit(X, y)
        lgb_preds_all.append(m_l.predict_proba(X_mat)[:, 1])

        m_x = xgb.XGBClassifier(**{**xgb_params, "random_state": seed})
        m_x.fit(X, y)
        xgb_preds_all.append(m_x.predict_proba(X_mat)[:, 1])

        m_c = CatBoostClassifier(**{**best_cb_params, "random_seed": seed})
        m_c.fit(X, y)
        cb_preds_all.append(m_c.predict_proba(X_mat)[:, 1])

        print(f"    Seed {seed} done")

    lgb_avg = np.mean(lgb_preds_all, axis=0)
    xgb_avg = np.mean(xgb_preds_all, axis=0)
    cb_avg = np.mean(cb_preds_all, axis=0)

    ensemble = weights[0] * lgb_avg + weights[1] * xgb_avg + weights[2] * cb_avg

    # Apply isotonic calibration if enabled
    if iso_model is not None:
        ensemble = iso_model.predict(ensemble)

    ensemble = np.clip(ensemble, CLIP_LOW, CLIP_HIGH)

    result = sample_sub.copy()
    result["Pred"] = ensemble
    return result


# Stage 1
print("\n  Stage 1 (past tournaments)...")
s1_result = predict_submission(sub1, m_feats, w_feats, feat_cols, SEEDS, hc_weights, final_iso)
s1_path = SUB_DIR / "submission_stage1_v3.csv"
s1_result.to_csv(s1_path, index=False)
print(f"  Saved: {s1_path}")
print(f"  Stats: mean={s1_result['Pred'].mean():.4f}, std={s1_result['Pred'].std():.4f}")
print(f"  Range: [{s1_result['Pred'].min():.4f}, {s1_result['Pred'].max():.4f}]")

# Stage 2
print("\n  Stage 2 (2026 predictions)...")
s2_result = predict_submission(sub2, m_feats, w_feats, feat_cols, SEEDS, hc_weights, final_iso)
s2_path = SUB_DIR / "submission_stage2_v3.csv"
s2_result.to_csv(s2_path, index=False)
print(f"  Saved: {s2_path}")
print(f"  Stats: mean={s2_result['Pred'].mean():.4f}, std={s2_result['Pred'].std():.4f}")
print(f"  Range: [{s2_result['Pred'].min():.4f}, {s2_result['Pred'].max():.4f}]")

# Save results
elapsed = time.time() - t0
results = {
    "version": "v3",
    "elapsed_seconds": elapsed,
    "elo_params": best_elo_params,
    "catboost_optuna_params": {k: round(v, 4) if isinstance(v, float) else v for k, v in best_cb_params.items()},
    "cv_brier": {
        "lgb": round(np.mean(lgb_scores), 5),
        "xgb": round(np.mean(xgb_scores), 5),
        "cb": round(np.mean(cb_scores), 5),
        "ensemble_avg": round(avg_brier, 5),
        "ensemble_hc": round(hc_brier, 5),
        "isotonic": round(np.mean(iso_scores), 5) if iso_scores else None,
    },
    "hc_weights": {"lgb": round(hc_weights[0], 3), "xgb": round(hc_weights[1], 3), "cb": round(hc_weights[2], 3)},
    "use_isotonic": bool(use_isotonic),
    "clip_range": [CLIP_LOW, CLIP_HIGH],
    "n_features": len(feat_cols),
    "n_optuna_trials": N_OPTUNA_TRIALS,
}

results_path = RES_DIR / "results_v3.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {results_path}")

print(f"\n{'=' * 60}")
print(f"DONE in {elapsed:.1f}s")
print(f"\nv3 IMPROVEMENTS:")
print(f"  [1] Brier Score evaluation (competition metric)")
print(f"  [2] Elo grid search: K={best_elo_params['k']}, home={best_elo_params['home_adv']}, carry={best_elo_params['carryover']}")
print(f"  [3] NaN-native handling (no fillna(0))")
print(f"  [4] Optuna CatBoost: {N_OPTUNA_TRIALS} trials -> Brier {study.best_value:.5f}")
print(f"  [5] Isotonic calibration: {'ENABLED' if use_isotonic else 'DISABLED (no improvement)'}")
print(f"  [6] Relaxed clipping: [{CLIP_LOW}, {CLIP_HIGH}]")
print(f"{'=' * 60}")
