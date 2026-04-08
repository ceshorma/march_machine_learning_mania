"""
March Machine Learning Mania 2026 - Improved v2
================================================
Improvements over v1 baseline:
1. External data: KenPom adjusted efficiency + Barttorvik BARTHAG
2. Massey ordinals (Men's only, NaN-filled for Women's)
3. 3-model ensemble: LightGBM + XGBoost + CatBoost
4. Multi-seed averaging (10 seeds)
5. Hill-climbing ensemble weights
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import time
import re

DATA_DIR = Path(__file__).parent.parent / "data"
EXT_DIR = Path("C:/Users/Admin/Desktop/march_data_temp")
SUB_DIR = Path(__file__).parent.parent / "submissions"
SUB_DIR.mkdir(exist_ok=True)

CLIP_LOW, CLIP_HIGH = 0.05, 0.95
SEEDS = [42, 2024, 2025, 1234, 5678, 7890, 3141, 2718, 1618, 4242]

# ── 0. Team Name Mapping ────────────────────────────────────────

# Map from external dataset team names -> Kaggle TeamName
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
}


def normalize_name(name):
    """Normalize a team name for matching."""
    name = str(name).strip()
    # Try direct mapping first
    if name in EXTERNAL_TO_KAGGLE:
        return EXTERNAL_TO_KAGGLE[name]
    # Strip trailing periods from "St." -> "St"
    name = re.sub(r'\.(?=\s|$)', '', name)
    return name


# ── 1. Load Data ─────────────────────────────────────────────────

print("=" * 60)
print("MARCH MACHINE LEARNING MANIA 2026 - IMPROVED v2")
print("=" * 60)

t0 = time.time()
print("\n[1/8] Loading data...")

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

print("\n[2/8] Loading external data (KenPom + Barttorvik)...")

# Build TeamName -> TeamID mapping
name_to_id = dict(zip(m_teams["TeamName"], m_teams["TeamID"]))

# KenPom Summary (1999-2025) - adjusted efficiency metrics
kenpom = pd.read_csv(EXT_DIR / "kenpom/INT _ KenPom _ Summary.csv")
kenpom["KaggleName"] = kenpom["TeamName"].apply(normalize_name)
kenpom["TeamID"] = kenpom["KaggleName"].map(name_to_id)

# KenPom Efficiency (has possession length)
kenpom_eff = pd.read_csv(EXT_DIR / "kenpom/INT _ KenPom _ Efficiency.csv")
kenpom_eff["KaggleName"] = kenpom_eff["Team"].apply(normalize_name)
kenpom_eff["TeamID"] = kenpom_eff["KaggleName"].map(name_to_id)

# Barttorvik combined (2013-2024)
bart_files = []
for yr in range(13, 26):
    f = EXT_DIR / f"barttorvik/cbb{yr}.csv"
    if f.exists():
        df = pd.read_csv(f)
        # Normalize column names
        col_map = {c: c.upper() for c in df.columns}
        if "TEAM" not in col_map.values():
            col_map = {c: c.strip().upper() for c in df.columns}
        df = df.rename(columns={c: c.strip().upper() for c in df.columns})
        if "YEAR" not in df.columns:
            df["YEAR"] = 2000 + yr
        bart_files.append(df)

bart_combined = pd.read_csv(EXT_DIR / "barttorvik/cbb.csv")
bart_combined.columns = [c.strip().upper() for c in bart_combined.columns]
bart_files.append(bart_combined)

bart = pd.concat(bart_files, ignore_index=True).drop_duplicates(subset=["TEAM", "YEAR"], keep="last")
bart["KaggleName"] = bart["TEAM"].apply(normalize_name)
bart["TeamID"] = bart["KaggleName"].map(name_to_id)

# Report mapping quality
kp_mapped = kenpom["TeamID"].notna().sum()
kp_total = len(kenpom)
bart_mapped = bart["TeamID"].notna().sum()
bart_total = len(bart)

print(f"  KenPom: {kp_mapped}/{kp_total} rows mapped ({kp_mapped/kp_total*100:.1f}%)")
print(f"  Barttorvik: {bart_mapped}/{bart_total} rows mapped ({bart_mapped/bart_total*100:.1f}%)")

# Show unmapped teams (for debugging)
kp_unmapped = kenpom[kenpom["TeamID"].isna()]["TeamName"].unique()
bart_unmapped = bart[bart["TeamID"].isna()]["TEAM"].unique()
if len(kp_unmapped) > 0:
    print(f"  KenPom unmapped ({len(kp_unmapped)}): {sorted(kp_unmapped)[:15]}...")
if len(bart_unmapped) > 0:
    print(f"  Barttorvik unmapped ({len(bart_unmapped)}): {sorted(bart_unmapped)[:15]}...")


# ── 3. Elo Ratings ──────────────────────────────────────────────

print("\n[3/8] Computing Elo ratings...")

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

m_elo = compute_elo(m_reg, m_conf)
w_elo = compute_elo(w_reg, w_conf)

top_names = {r.TeamID: r.TeamName for _, r in m_teams.iterrows()}
latest = max(s for s, _ in m_elo.keys())
top = sorted([(t, e) for (s, t), e in m_elo.items() if s == latest], key=lambda x: -x[1])[:10]
print(f"  Top 10 Men's Elo ({latest}):")
for tid, elo in top:
    print(f"    {top_names.get(tid, tid):25s} {elo:.0f}")


# ── 4. Efficiency from Detailed Results ─────────────────────────

print("\n[4/8] Computing efficiency metrics...")

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
print(f"  Men's efficiency: {m_eff.shape}")
print(f"  Women's efficiency: {w_eff.shape}")


# ── 5. Seed + Massey Features ──────────────────────────────────

print("\n[5/8] Building seed & Massey features...")

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
print(f"  Massey aggregated: {m_massey_agg.shape}")


# ── 6. Build Team Features (with External Data) ────────────────

print("\n[6/8] Building team features with external data...")

def build_team_feats(elo_dict, eff_df, seeds_df, massey_df, conf_df, reg_df,
                     kenpom_df=None, bart_df=None, max_day=132):
    """Build per-team-per-season feature vector with external data."""
    teams_by_season = set()
    for _, g in reg_df[reg_df.DayNum <= max_day].iterrows():
        teams_by_season.add((int(g.Season), int(g.WTeamID)))
        teams_by_season.add((int(g.Season), int(g.LTeamID)))

    rows = []
    for season, tid in teams_by_season:
        row = {"Season": season, "TeamID": tid}

        # Elo
        row["Elo"] = elo_dict.get((season, tid), 1500)

        # Our efficiency (from box scores)
        ef = eff_df[(eff_df.Season == season) & (eff_df.TeamID == tid)]
        if len(ef) > 0:
            ef = ef.iloc[0]
            for c in ["OffEff", "DefEff", "NetEff", "eFG", "TORate", "ORBRate", "FTRate", "WinPct"]:
                row[c] = ef[c]

        # Seed
        sd = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == tid)]
        row["SeedNum"] = sd["SeedNum"].values[0] if len(sd) > 0 else 8.5

        # Massey ordinals (Men's only)
        if massey_df is not None and len(massey_df) > 0:
            ms = massey_df[(massey_df.Season == season) & (massey_df.TeamID == tid)]
            if len(ms) > 0:
                ms = ms.iloc[0]
                for c in ["MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax", "MasseyStd"]:
                    row[c] = ms[c]

        # KenPom (external, Men's only)
        if kenpom_df is not None:
            kp = kenpom_df[(kenpom_df.Season == season) & (kenpom_df.TeamID == tid)]
            if len(kp) > 0:
                kp = kp.iloc[0]
                row["KP_AdjOE"] = kp.get("AdjOE", np.nan) if "AdjOE" in kp.index else np.nan
                row["KP_AdjDE"] = kp.get("AdjDE", np.nan) if "AdjDE" in kp.index else np.nan
                row["KP_AdjEM"] = kp.get("AdjEM", np.nan) if "AdjEM" in kp.index else np.nan
                row["KP_AdjTempo"] = kp.get("AdjTempo", np.nan) if "AdjTempo" in kp.index else np.nan

        # Barttorvik (external, Men's only)
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
        row["RecordGames"] = total

        rows.append(row)

    return pd.DataFrame(rows)


# Prepare KenPom for merging: rename columns
kenpom_for_merge = kenpom.rename(columns={
    "AdjTempo": "AdjTempo", "RankAdjEM": "RankAdjEM",
}).copy()
# Ensure AdjOE, AdjDE, AdjEM exist
col_renames = {}
for c in kenpom_for_merge.columns:
    if "adjoe" in c.lower() or ("adj" in c.lower() and "oe" in c.lower()):
        col_renames[c] = "AdjOE"
    elif "adjde" in c.lower() or ("adj" in c.lower() and "de" in c.lower()):
        col_renames[c] = "AdjDE"

# Check what columns KenPom Summary actually has
kp_summary_cols = kenpom.columns.tolist()
print(f"  KenPom Summary columns: {kp_summary_cols}")

# Map: KenPom Summary has OE/DE/AdjOE style or different names
# Let's check and rename properly
kp_rename = {}
for c in kp_summary_cols:
    cl = c.lower()
    if cl == "adjoe" or (cl.startswith("adj") and "offen" in cl):
        kp_rename[c] = "AdjOE"
    elif cl == "adjde" or (cl.startswith("adj") and "defen" in cl):
        kp_rename[c] = "AdjDE"
    elif cl == "adjem" or (cl.startswith("adj") and "marg" in cl) or (cl.startswith("adj") and "em" in cl):
        kp_rename[c] = "AdjEM"
    elif cl == "adjtempo" or (cl.startswith("adj") and "tempo" in cl):
        kp_rename[c] = "AdjTempo"

# Use the KenPom Efficiency file which has explicit column names
kp_eff_cols = kenpom_eff.columns.tolist()
print(f"  KenPom Efficiency columns: {kp_eff_cols}")

# Build a clean KenPom merge table
kp_clean = kenpom_eff[["Season", "TeamID"]].copy()
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
print(f"  KenPom clean merge table: {kp_clean.shape}")
print(f"  KenPom columns: {kp_clean.columns.tolist()}")

# Clean Barttorvik
bart_clean = bart[["YEAR", "TeamID"]].copy()
for src, dst in [("BARTHAG", "BARTHAG"), ("ADJOE", "ADJOE"), ("ADJDE", "ADJDE"),
                 ("EFG_O", "EFG_O"), ("EFG_D", "EFG_D"), ("WAB", "WAB"), ("ADJ_T", "ADJ_T")]:
    if src in bart.columns:
        bart_clean[dst] = bart[src].values
bart_clean = bart_clean.dropna(subset=["TeamID"]).copy()
bart_clean["TeamID"] = bart_clean["TeamID"].astype(int)
print(f"  Barttorvik clean: {bart_clean.shape}")

# Build team features
m_feats = build_team_feats(
    m_elo, m_eff, m_seeds, m_massey_agg, m_conf, m_reg,
    kenpom_df=kp_clean, bart_df=bart_clean
)
w_feats = build_team_feats(
    w_elo, w_eff, w_seeds, None, w_conf, w_reg,
    kenpom_df=None, bart_df=None
)

print(f"  Men's features: {m_feats.shape}")
print(f"  Women's features: {w_feats.shape}")
print(f"  Men's columns: {m_feats.columns.tolist()}")
print(f"  Women's columns: {w_feats.columns.tolist()}")


# ── 7. Build Matchup Training Data ─────────────────────────────

print("\n[7/8] Building matchup training data & evaluating...")

def make_matchup(team_feats, tourney_df, feat_cols_list):
    """Create matchup training data from tournament results."""
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
        for c in feat_cols_list:
            v1 = f1.get(c, np.nan)
            v2 = f2.get(c, np.nan)
            row[f"{c}_diff"] = (v1 - v2) if pd.notna(v1) and pd.notna(v2) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# Feature columns for matchup differences
exclude = {"Season", "TeamID", "Games", "RecordGames"}
m_feat_cols = [c for c in m_feats.columns if c not in exclude and m_feats[c].dtype in [np.float64, np.int64, np.float32]]
w_feat_cols = [c for c in w_feats.columns if c not in exclude and w_feats[c].dtype in [np.float64, np.int64, np.float32]]

m_train = make_matchup(m_feats, m_tourney, m_feat_cols)
m_train["is_mens"] = 1
w_train = make_matchup(w_feats, w_tourney, w_feat_cols)
w_train["is_mens"] = 0

# Align columns: fill missing with NaN
common = sorted(set(m_train.columns) & set(w_train.columns))
all_cols = sorted(set(m_train.columns) | set(w_train.columns))

for c in all_cols:
    if c not in m_train.columns:
        m_train[c] = np.nan
    if c not in w_train.columns:
        w_train[c] = np.nan

train = pd.concat([m_train[all_cols], w_train[all_cols]], ignore_index=True)
train = train.fillna(0)

meta = ["Season", "TeamID1", "TeamID2", "target"]
feat_cols = sorted([c for c in train.columns if c not in meta])

X = train[feat_cols].values
y = train["target"].values
seasons = train["Season"]

print(f"  Combined training: {train.shape}")
print(f"  Features ({len(feat_cols)}): {feat_cols}")
print(f"  Target mean: {y.mean():.3f}")
print(f"  Seasons: {int(seasons.min())} - {int(seasons.max())}")


# ── Temporal CV ─────────────────────────────────────────────────

all_seasons = sorted(seasons.unique())

# Seed baseline
seed_col = feat_cols.index("SeedNum_diff") if "SeedNum_diff" in feat_cols else None
if seed_col is not None:
    seed_scores = []
    for val_season in all_seasons[-10:]:
        tr_mask = seasons < val_season
        va_mask = seasons == val_season
        if va_mask.sum() == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X[tr_mask][:, [seed_col]], y[tr_mask])
        pred = np.clip(lr.predict_proba(X[va_mask][:, [seed_col]])[:, 1], CLIP_LOW, CLIP_HIGH)
        score = log_loss(y[va_mask], pred)
        seed_scores.append(score)
        print(f"  Seed baseline | {int(val_season)} | n={int(va_mask.sum()):3d} | LogLoss: {score:.5f}")
    print(f"  Seed baseline MEAN: {np.mean(seed_scores):.5f}\n")

# LightGBM
lgb_params = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "n_estimators": 500,
    "verbosity": -1, "n_jobs": -1,
}

# Collect OOF predictions for ensemble weight optimization
lgb_oof = np.full(len(y), np.nan)
xgb_oof = np.full(len(y), np.nan)
cb_oof = np.full(len(y), np.nan)

lgb_scores, xgb_scores, cb_scores = [], [], []

xgb_params = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "n_estimators": 500,
    "tree_method": "hist", "verbosity": 0, "n_jobs": -1,
}

cb_params = {
    "loss_function": "Logloss", "eval_metric": "Logloss",
    "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0,
    "iterations": 500, "verbose": 0,
}

print("  Running 3-model temporal CV...")
for val_season in all_seasons[-10:]:
    tr_mask = seasons < val_season
    va_mask = seasons == val_season
    if va_mask.sum() == 0:
        continue

    X_tr, X_va = X[tr_mask], X[va_mask]
    y_tr, y_va = y[tr_mask], y[va_mask]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    m_lgb.fit(X_tr, y_tr)
    p_lgb = np.clip(m_lgb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    lgb_oof[va_mask] = p_lgb
    s_lgb = log_loss(y_va, p_lgb)
    lgb_scores.append(s_lgb)

    # XGBoost
    m_xgb = xgb.XGBClassifier(**{**xgb_params, "random_state": 42})
    m_xgb.fit(X_tr, y_tr)
    p_xgb = np.clip(m_xgb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    xgb_oof[va_mask] = p_xgb
    s_xgb = log_loss(y_va, p_xgb)
    xgb_scores.append(s_xgb)

    # CatBoost
    m_cb = CatBoostClassifier(**{**cb_params, "random_seed": 42})
    m_cb.fit(X_tr, y_tr)
    p_cb = np.clip(m_cb.predict_proba(X_va)[:, 1], CLIP_LOW, CLIP_HIGH)
    cb_oof[va_mask] = p_cb
    s_cb = log_loss(y_va, p_cb)
    cb_scores.append(s_cb)

    print(f"  {int(val_season)} | LGB: {s_lgb:.5f} | XGB: {s_xgb:.5f} | CB: {s_cb:.5f}")

print(f"\n  LightGBM MEAN: {np.mean(lgb_scores):.5f}")
print(f"  XGBoost  MEAN: {np.mean(xgb_scores):.5f}")
print(f"  CatBoost MEAN: {np.mean(cb_scores):.5f}")

# Simple average ensemble on OOF
valid = ~(np.isnan(lgb_oof) | np.isnan(xgb_oof) | np.isnan(cb_oof))
avg_oof = (lgb_oof[valid] + xgb_oof[valid] + cb_oof[valid]) / 3
avg_score = log_loss(y[valid], np.clip(avg_oof, CLIP_LOW, CLIP_HIGH))
print(f"  Ensemble AVG   MEAN: {avg_score:.5f}")

# Hill climbing weights
def hill_climbing(preds_list, y_true, n_iter=300, step=0.02):
    n = len(preds_list)
    weights = [1/n] * n
    best_score = log_loss(y_true, np.clip(sum(w*p for w, p in zip(weights, preds_list)), CLIP_LOW, CLIP_HIGH))

    for _ in range(n_iter):
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
                new_w = [w/s for w in new_w]
                pred = np.clip(sum(w*p for w, p in zip(new_w, preds_list)), CLIP_LOW, CLIP_HIGH)
                score = log_loss(y_true, pred)
                if score < best_score:
                    weights = new_w
                    best_score = score

    return weights, best_score

hc_weights, hc_score = hill_climbing(
    [lgb_oof[valid], xgb_oof[valid], cb_oof[valid]], y[valid]
)
print(f"  Ensemble HC    MEAN: {hc_score:.5f}")
print(f"  Weights: LGB={hc_weights[0]:.3f}, XGB={hc_weights[1]:.3f}, CB={hc_weights[2]:.3f}")

# Feature importance
model_full = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
model_full.fit(X, y)
imp = sorted(zip(feat_cols, model_full.feature_importances_), key=lambda x: -x[1])
print(f"\n  Top 15 features:")
for name, val in imp[:15]:
    print(f"    {name:35s} {val:5d}")


# ── 8. Generate Submissions ────────────────────────────────────

print("\n[8/8] Generating submissions with 3-model ensemble...")

def predict_submission(sample_sub, m_feats, w_feats, feat_cols, seeds_list, weights):
    """Generate predictions with 3-model multi-seed ensemble."""
    all_feats = pd.concat([m_feats, w_feats], ignore_index=True)
    fc = set()
    for c in feat_cols:
        if c.endswith("_diff"):
            fc.add(c.replace("_diff", ""))
        elif c == "is_mens":
            pass

    # Build feature matrix
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
                row[f"{c}_diff"] = (v1 - v2) if pd.notna(v1) and pd.notna(v2) else 0
            row["is_mens"] = 1 if t1 < 3000 else 0
        else:
            for c in fc:
                row[f"{c}_diff"] = 0
            row["is_mens"] = 1 if t1 < 3000 else 0
        rows.append(row)

    X_sub = pd.DataFrame(rows)
    ids = X_sub["ID"].values

    # Ensure same columns
    for c in feat_cols:
        if c not in X_sub.columns:
            X_sub[c] = 0
    X_mat = X_sub[feat_cols].fillna(0).values

    # Multi-seed 3-model prediction
    lgb_preds_all = []
    xgb_preds_all = []
    cb_preds_all = []

    for seed in seeds_list:
        m_l = lgb.LGBMClassifier(**{**lgb_params, "random_state": seed})
        m_l.fit(X, y)
        lgb_preds_all.append(m_l.predict_proba(X_mat)[:, 1])

        m_x = xgb.XGBClassifier(**{**xgb_params, "random_state": seed})
        m_x.fit(X, y)
        xgb_preds_all.append(m_x.predict_proba(X_mat)[:, 1])

        m_c = CatBoostClassifier(**{**cb_params, "random_seed": seed})
        m_c.fit(X, y)
        cb_preds_all.append(m_c.predict_proba(X_mat)[:, 1])

        print(f"    Seed {seed} done")

    lgb_avg = np.mean(lgb_preds_all, axis=0)
    xgb_avg = np.mean(xgb_preds_all, axis=0)
    cb_avg = np.mean(cb_preds_all, axis=0)

    # Weighted ensemble
    ensemble = weights[0] * lgb_avg + weights[1] * xgb_avg + weights[2] * cb_avg
    ensemble = np.clip(ensemble, CLIP_LOW, CLIP_HIGH)

    result = sample_sub.copy()
    result["Pred"] = ensemble
    return result


# Stage 1
print("  Stage 1 (2022-2025 validation)...")
s1_result = predict_submission(sub1, m_feats, w_feats, feat_cols, SEEDS, hc_weights)
s1_path = SUB_DIR / "submission_stage1_v2.csv"
s1_result.to_csv(s1_path, index=False)
print(f"  Saved: {s1_path}")
print(f"  Stats: mean={s1_result['Pred'].mean():.4f}, std={s1_result['Pred'].std():.4f}")

# Stage 2
print("\n  Stage 2 (2026 predictions)...")
s2_result = predict_submission(sub2, m_feats, w_feats, feat_cols, SEEDS, hc_weights)
s2_path = SUB_DIR / "submission_stage2_v2.csv"
s2_result.to_csv(s2_path, index=False)
print(f"  Saved: {s2_path}")
print(f"  Stats: mean={s2_result['Pred'].mean():.4f}, std={s2_result['Pred'].std():.4f}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"DONE in {elapsed:.1f}s")
print(f"Improvements over v1:")
print(f"  + KenPom adjusted efficiency (AdjOE, AdjDE, AdjEM, AdjTempo)")
print(f"  + Barttorvik BARTHAG, WAB, adjusted metrics")
print(f"  + Massey ordinals for Men's")
print(f"  + 3-model ensemble (LGB + XGB + CB) with hill climbing weights")
print(f"  + 10-seed averaging (vs 5 in v1)")
print(f"{'=' * 60}")
