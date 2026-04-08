"""
Modular Experiment Runner for March Machine Learning Mania 2026.

Usage: python notebooks/run_experiment.py experiments/exp_NNN/config.json

Reads config JSON, runs the full pipeline, saves results + log,
and updates the experiment registry.
"""

import sys
import json
import time
import io
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.external_data import (
    load_kenpom_efficiency, load_barttorvik_individual,
    load_kp_bt_combined, load_resumes, load_evan_miya,
    load_538_ratings, load_coach_features, normalize_name,
)
from utils.features import build_conference_features

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXT_DIR = Path("C:/Users/Admin/Desktop/march_data_temp")
SUB_DIR = BASE_DIR / "submissions"
EXP_DIR = BASE_DIR / "experiments"
SUB_DIR.mkdir(exist_ok=True)

SEEDS_LIST = [42, 2024, 2025, 1234, 5678, 7890, 3141, 2718, 1618, 4242]


# ── TeeLogger ───────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ── Elo System (inline, proven from v3b) ────────────────────────

def compute_bt_strengths(reg_df, carryover=0.65, l2_reg=0.01):
    """Compute Bradley-Terry team strengths per season via MLE (scipy)."""
    from scipy.optimize import minimize as sp_minimize
    strengths = {}
    seasons = sorted(reg_df["Season"].unique())

    for season in seasons:
        sg = reg_df[reg_df.Season == season]
        teams = sorted(set(sg["WTeamID"]) | set(sg["LTeamID"]))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        theta0 = np.zeros(n + 1)
        for i, t in enumerate(teams):
            prev = strengths.get((season - 1, t), 0.0)
            theta0[i] = carryover * prev
        theta0[-1] = 0.3

        w_idx, l_idx, home_flags = [], [], []
        for _, g in sg.iterrows():
            wi = team_to_idx.get(int(g.WTeamID))
            li = team_to_idx.get(int(g.LTeamID))
            if wi is not None and li is not None:
                w_idx.append(wi)
                l_idx.append(li)
                home_flags.append(1 if g.WLoc == "H" else (-1 if g.WLoc == "A" else 0))

        w_idx = np.array(w_idx)
        l_idx = np.array(l_idx)
        home_flags = np.array(home_flags, dtype=np.float64)

        def neg_log_lik(x):
            theta, h = x[:n], x[n]
            diff = theta[w_idx] - theta[l_idx] + h * home_flags
            return np.sum(np.logaddexp(0, -diff)) + l2_reg * np.sum(theta ** 2)

        def grad(x):
            theta, h = x[:n], x[n]
            diff = theta[w_idx] - theta[l_idx] + h * home_flags
            sig = 1.0 / (1.0 + np.exp(diff))
            g = np.zeros(n + 1)
            np.add.at(g[:n], w_idx, -sig)
            np.add.at(g[:n], l_idx, sig)
            g[:n] += 2 * l2_reg * theta
            g[n] = np.sum(-sig * home_flags)
            return g

        res = sp_minimize(neg_log_lik, theta0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10})
        theta_opt = res.x[:n]
        theta_opt -= theta_opt.mean()

        for i, t in enumerate(teams):
            strengths[(season, t)] = theta_opt[i]

    return strengths


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


def quick_elo_eval(elo_dict, seeds_df, tourney_df, clip_lo, clip_hi, eval_seasons=5):
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
    eval_szns = sorted(df["Season"].unique())[-eval_seasons:]
    scores = []
    for vs in eval_szns:
        tr, va = df[df.Season < vs], df[df.Season == vs]
        if len(va) == 0:
            continue
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(tr[["elo_diff", "seed_diff"]].values, tr["target"].values)
        pred = np.clip(lr.predict_proba(va[["elo_diff", "seed_diff"]].values)[:, 1], clip_lo, clip_hi)
        scores.append(brier_score_loss(va["target"].values, pred))
    return np.mean(scores) if scores else 1.0


# ── Efficiency (inline, proven from v3b) ────────────────────────

def compute_efficiency(det_df, max_day=132, last_n=0,
                       consistency=False, close_games=False,
                       neutral_court=False, ast_to=False,
                       eff_ratio=False, overtime=False):
    reg = det_df[det_df.DayNum <= max_day].copy()
    loc_map = {"H": "A", "A": "H", "N": "N"}
    rows = []
    for _, g in reg.iterrows():
        s = g.Season
        wloc = g.get("WLoc", "N")
        for prefix, opp_prefix, tid, oid, won, score, opp_score, loc in [
            ("W", "L", g.WTeamID, g.LTeamID, 1, g.WScore, g.LScore, wloc),
            ("L", "W", g.LTeamID, g.WTeamID, 0, g.LScore, g.WScore, loc_map.get(wloc, "N")),
        ]:
            fga, orb, to, fta = g[f"{prefix}FGA"], g[f"{prefix}OR"], g[f"{prefix}TO"], g[f"{prefix}FTA"]
            fgm, fgm3, opp_drb = g[f"{prefix}FGM"], g[f"{prefix}FGM3"], g[f"{opp_prefix}DR"]
            poss = max(fga - orb + to + 0.475 * fta, 1)
            opp_poss = max(g[f"{opp_prefix}FGA"] - g[f"{opp_prefix}OR"] + g[f"{opp_prefix}TO"] + 0.475 * g[f"{opp_prefix}FTA"], 1)
            off_eff = score / poss * 100
            def_eff = opp_score / opp_poss * 100
            margin = score - opp_score
            row = {
                "Season": s, "TeamID": int(tid), "DayNum": g.DayNum,
                "OffEff": off_eff, "DefEff": def_eff,
                "eFG": (fgm + 0.5 * fgm3) / max(fga, 1), "TORate": to / poss,
                "ORBRate": orb / max(orb + opp_drb, 1), "FTRate": fta / max(fga, 1),
                "Won": won, "Margin": margin, "Loc": loc,
            }
            if ast_to:
                ast = g[f"{prefix}Ast"]
                stl = g[f"{prefix}Stl"]
                blk = g[f"{prefix}Blk"]
                row["AstToTO"] = ast / max(to, 1)
                row["StealRate"] = stl / poss
                row["BlockRate"] = blk / poss
            if overtime and "NumOT" in g.index:
                row["IsOT"] = 1 if g["NumOT"] > 0 else 0
                row["OTWon"] = 1 if (g["NumOT"] > 0 and won) else 0
            rows.append(row)
    gdf = pd.DataFrame(rows)
    agg = gdf.groupby(["Season", "TeamID"]).agg(
        OffEff=("OffEff", "mean"), DefEff=("DefEff", "mean"),
        eFG=("eFG", "mean"), TORate=("TORate", "mean"),
        ORBRate=("ORBRate", "mean"), FTRate=("FTRate", "mean"),
        WinPct=("Won", "mean"), Games=("Won", "count"),
    ).reset_index()
    agg["NetEff"] = agg["OffEff"] - agg["DefEff"]

    # Overtime features (optional)
    if overtime and "IsOT" in gdf.columns:
        ot_agg = gdf.groupby(["Season", "TeamID"]).agg(
            OTGames=("IsOT", "sum"), OTWinRate=("OTWon", lambda x: x.sum() / max(x.count(), 1)),
        ).reset_index().fillna(0)
        agg = agg.merge(ot_agg, on=["Season", "TeamID"], how="left")
        agg["OTGames"] = agg["OTGames"].fillna(0)
        agg["OTWinRate"] = agg["OTWinRate"].fillna(0)

    # Efficiency ratio (optional): OffEff / DefEff captures dominance as ratio
    if eff_ratio:
        agg["EffRatio"] = agg["OffEff"] / agg["DefEff"].replace(0, 1)

    # Consistency features: std dev of per-game performance
    if consistency:
        cons = gdf.groupby(["Season", "TeamID"]).agg(
            OffEff_Std=("OffEff", "std"), DefEff_Std=("DefEff", "std"),
            Margin_Std=("Margin", "std"),
        ).reset_index().fillna(0)
        agg = agg.merge(cons, on=["Season", "TeamID"], how="left")

    # Close game features: performance in games decided by <=5 points
    if close_games:
        close = gdf[gdf["Margin"].abs() <= 5].copy()
        if len(close) > 0:
            cg = close.groupby(["Season", "TeamID"]).agg(
                CloseGameWR=("Won", "mean"), CloseGameN=("Won", "count"),
            ).reset_index()
            agg = agg.merge(cg, on=["Season", "TeamID"], how="left")
            agg["CloseGameWR"] = agg["CloseGameWR"].fillna(0.5)
            agg["CloseGameN"] = agg["CloseGameN"].fillna(0)

    # Neutral court features: performance on neutral courts
    if neutral_court:
        neutral = gdf[gdf["Loc"] == "N"]
        if len(neutral) > 0:
            nc = neutral.groupby(["Season", "TeamID"]).agg(
                NeutralWR=("Won", "mean"),
            ).reset_index()
            agg = agg.merge(nc, on=["Season", "TeamID"], how="left")
            agg["NeutralWR"] = agg["NeutralWR"].fillna(agg["WinPct"])
        # Home court dependency: OffEff at home - OffEff away
        home_eff = gdf[gdf["Loc"] == "H"].groupby(["Season", "TeamID"])["OffEff"].mean().reset_index().rename(columns={"OffEff": "_home"})
        away_eff = gdf[gdf["Loc"] == "A"].groupby(["Season", "TeamID"])["OffEff"].mean().reset_index().rename(columns={"OffEff": "_away"})
        hcd = home_eff.merge(away_eff, on=["Season", "TeamID"], how="outer")
        hcd["HomeCourtDep"] = hcd["_home"].fillna(0) - hcd["_away"].fillna(0)
        agg = agg.merge(hcd[["Season", "TeamID", "HomeCourtDep"]], on=["Season", "TeamID"], how="left")
        agg["HomeCourtDep"] = agg["HomeCourtDep"].fillna(0)

    # Assist-to-Turnover and defensive stats
    if ast_to:
        ato = gdf.groupby(["Season", "TeamID"]).agg(
            AstToTO=("AstToTO", "mean"), StealRate=("StealRate", "mean"),
            BlockRate=("BlockRate", "mean"),
        ).reset_index()
        agg = agg.merge(ato, on=["Season", "TeamID"], how="left")

    # Momentum: last N games
    if last_n > 0:
        gdf_sorted = gdf.sort_values(["Season", "TeamID", "DayNum"])
        last = gdf_sorted.groupby(["Season", "TeamID"]).tail(last_n)
        mom = last.groupby(["Season", "TeamID"]).agg(
            OffEff_LN=("OffEff", "mean"), DefEff_LN=("DefEff", "mean"),
            eFG_LN=("eFG", "mean"), WinPct_LN=("Won", "mean"),
        ).reset_index()
        mom["NetEff_LN"] = mom["OffEff_LN"] - mom["DefEff_LN"]
        agg = agg.merge(mom, on=["Season", "TeamID"], how="left")
        # Momentum delta: recent form vs season average
        agg["MomDelta_Off"] = agg["OffEff_LN"] - agg["OffEff"]
        agg["MomDelta_Def"] = agg["DefEff_LN"] - agg["DefEff"]
        agg["MomDelta_Win"] = agg["WinPct_LN"] - agg["WinPct"]

    return agg


def compute_conf_tourney(conf_tourney_df):
    """Conference tournament performance. Leak-free (DayNum 115-132)."""
    wins = conf_tourney_df.groupby(["Season", "WTeamID"]).size().reset_index(name="ConfTourneyW")
    wins.rename(columns={"WTeamID": "TeamID"}, inplace=True)
    losses = conf_tourney_df.groupby(["Season", "LTeamID"]).size().reset_index(name="ConfTourneyL")
    losses.rename(columns={"LTeamID": "TeamID"}, inplace=True)
    ct = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
    ct["ConfTourneyWR"] = ct["ConfTourneyW"] / (ct["ConfTourneyW"] + ct["ConfTourneyL"])
    # Identify champion: WTeamID of last game per (Season, ConfAbbrev)
    last_games = conf_tourney_df.sort_values("DayNum").groupby(["Season", "ConfAbbrev"]).last().reset_index()
    champs = last_games[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    champs["IsConfTourneyChamp"] = 1
    ct = ct.merge(champs, on=["Season", "TeamID"], how="left")
    ct["IsConfTourneyChamp"] = ct["IsConfTourneyChamp"].fillna(0)
    return ct[["Season", "TeamID", "ConfTourneyWR", "IsConfTourneyChamp"]]


# ── Massey Aggregation ──────────────────────────────────────────

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


# ── Build Team Features ─────────────────────────────────────────

def compute_rest_days(reg_df, tourney_df):
    """Compute days of rest before tournament for each team-season."""
    rest = {}
    for season in sorted(tourney_df["Season"].unique()):
        st = tourney_df[tourney_df.Season == season]
        tourney_teams = set(st["WTeamID"].astype(int)) | set(st["LTeamID"].astype(int))
        first_tourney_day = st["DayNum"].min()
        sr = reg_df[reg_df.Season == season]
        for tid in tourney_teams:
            team_games = sr[(sr.WTeamID == tid) | (sr.LTeamID == tid)]
            if len(team_games) > 0:
                rest[(season, tid)] = int(first_tourney_day - team_games["DayNum"].max())
            else:
                rest[(season, tid)] = 7
    return rest


def build_team_feats(elo_dict, eff_df, seeds_df, massey_df, reg_df,
                     ext_dfs=None, max_day=132, bt_dict=None, rest_dict=None, travel_dict=None):
    """Build per-team-season features. ext_dfs is a dict of name -> DataFrame."""
    if ext_dfs is None:
        ext_dfs = {}

    teams_by_season = set()
    for _, g in reg_df[reg_df.DayNum <= max_day].iterrows():
        teams_by_season.add((int(g.Season), int(g.WTeamID)))
        teams_by_season.add((int(g.Season), int(g.LTeamID)))

    rows = []
    for season, tid in teams_by_season:
        row = {"Season": season, "TeamID": tid}
        row["Elo"] = elo_dict.get((season, tid), 1500)
        if bt_dict is not None:
            row["BT_Strength"] = bt_dict.get((season, tid), 0.0)
        if rest_dict is not None:
            row["RestDays"] = rest_dict.get((season, tid), 7.0)
        if travel_dict is not None:
            row["TravelDist"] = travel_dict.get((season, tid), 0.0)

        # Efficiency (+ momentum columns if present)
        ef = eff_df[(eff_df.Season == season) & (eff_df.TeamID == tid)]
        if len(ef) > 0:
            ef = ef.iloc[0]
            for c in ef.index:
                if c not in ("Season", "TeamID", "Games", "DayNum"):
                    row[c] = ef[c]

        # Seed
        sd = seeds_df[(seeds_df.Season == season) & (seeds_df.TeamID == tid)]
        row["SeedNum"] = sd["SeedNum"].values[0] if len(sd) > 0 else 8.5

        # Massey
        if massey_df is not None and len(massey_df) > 0:
            ms = massey_df[(massey_df.Season == season) & (massey_df.TeamID == tid)]
            if len(ms) > 0:
                ms = ms.iloc[0]
                for c in ["MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax", "MasseyStd"]:
                    if c in ms.index:
                        row[c] = ms[c]

        # External data (merge by Season/TeamID)
        for name, edf in ext_dfs.items():
            season_col = "Season" if "Season" in edf.columns else "YEAR"
            if season_col not in edf.columns:
                continue
            match = edf[(edf[season_col] == season) & (edf["TeamID"] == tid)]
            if len(match) > 0:
                m = match.iloc[0]
                for c in edf.columns:
                    if c not in ("Season", "YEAR", "TeamID"):
                        row[c] = m[c]

        # Win/Loss record
        wins = len(reg_df[(reg_df.Season == season) & (reg_df.WTeamID == tid) & (reg_df.DayNum <= max_day)])
        losses = len(reg_df[(reg_df.Season == season) & (reg_df.LTeamID == tid) & (reg_df.DayNum <= max_day)])
        row["RecordWinPct"] = wins / max(wins + losses, 1)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Matchup Builder ─────────────────────────────────────────────

def make_matchup(team_feats, tourney_df, feat_cols):
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
        for c in feat_cols:
            v1, v2 = f1.get(c, np.nan), f2.get(c, np.nan)
            row[f"{c}_diff"] = (v1 - v2) if (pd.notna(v1) and pd.notna(v2)) else 0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Hill Climbing (Brier Score) ─────────────────────────────────

def hill_climbing(preds_list, y_true, clip_lo, clip_hi, n_iter=500, step=0.01):
    n = len(preds_list)
    weights = [1 / n] * n
    best_score = brier_score_loss(y_true, np.clip(sum(w * p for w, p in zip(weights, preds_list)), clip_lo, clip_hi))
    for _ in range(n_iter):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                nw = weights.copy()
                nw[i] = max(0, nw[i] + step)
                nw[j] = max(0, nw[j] - step)
                s = sum(nw)
                if s == 0:
                    continue
                nw = [w / s for w in nw]
                pred = np.clip(sum(w * p for w, p in zip(nw, preds_list)), clip_lo, clip_hi)
                score = brier_score_loss(y_true, pred)
                if score < best_score:
                    weights, best_score = nw, score
                    improved = True
        if not improved:
            break
    return weights, best_score


# ── Submission Predictor ────────────────────────────────────────

def predict_submission(sample_sub, m_feats, w_feats, feat_cols, seeds, weights,
                       X_train, y_train, lgb_params, xgb_params, cb_params, clip_lo, clip_hi,
                       seed_hist_data=None, lr_C=1.0, cb_regression=False,
                       iso_calibration_data=None):
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
                row[f"{c}_diff"] = (v1 - v2) if (pd.notna(v1) and pd.notna(v2)) else 0
            row["is_mens"] = 1 if t1 < 3000 else 0
        else:
            for c in fc:
                row[f"{c}_diff"] = 0
            row["is_mens"] = 1 if t1 < 3000 else 0
        rows.append(row)

    X_sub = pd.DataFrame(rows)
    ids = X_sub["ID"].values

    # Add SeedHistWR for submissions if needed
    if seed_hist_data is not None and "SeedHistWR" in feat_cols:
        all_rates = seed_hist_data[0]
        seed_map_fn = seed_hist_data[1]

        def _compute_wr(rates_dict, s, t1, t2):
            s1 = seed_map_fn.get((s, t1), 8)
            s2 = seed_map_fn.get((s, t2), 8)
            strong, weak = min(s1, s2), max(s1, s2)
            if strong == weak:
                return 0.5
            rate = rates_dict.get((strong, weak), 0.5)
            return rate if s1 <= s2 else 1.0 - rate

        wr_vals = []
        for _, r in X_sub.iterrows():
            parts = r["ID"].split("_")
            s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])
            wr_vals.append(_compute_wr(all_rates, s, t1, t2))
        X_sub["SeedHistWR"] = wr_vals

        # Gender-specific SeedHistWR for submissions
        if len(seed_hist_data) == 4:
            all_rates_m, all_rates_w = seed_hist_data[2], seed_hist_data[3]
            for col_name, gender_rates, is_mens_val in [
                ("SeedHistWR_M", all_rates_m, 1), ("SeedHistWR_W", all_rates_w, 0)
            ]:
                if col_name in feat_cols:
                    g_vals = []
                    for _, r in X_sub.iterrows():
                        parts = r["ID"].split("_")
                        s, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])
                        is_m = 1 if t1 < 3000 else 0
                        if is_m == is_mens_val:
                            g_vals.append(_compute_wr(gender_rates, s, t1, t2))
                        else:
                            g_vals.append(0.5)
                    X_sub[col_name] = g_vals

    # Interaction features for submissions
    if "Elo_x_SeedHist" in feat_cols and "Elo_diff" in X_sub.columns and "SeedHistWR" in X_sub.columns:
        X_sub["Elo_x_SeedHist"] = X_sub["Elo_diff"] * X_sub["SeedHistWR"]
    if "Elo_x_Seed" in feat_cols and "Elo_diff" in X_sub.columns and "SeedNum_diff" in X_sub.columns:
        X_sub["Elo_x_Seed"] = X_sub["Elo_diff"] * X_sub["SeedNum_diff"]
    if "SeedHist_x_Gender" in feat_cols and "SeedHistWR" in X_sub.columns and "is_mens" in X_sub.columns:
        X_sub["SeedHist_x_Gender"] = X_sub["SeedHistWR"] * X_sub["is_mens"]

    for c in feat_cols:
        if c not in X_sub.columns:
            X_sub[c] = 0
    X_mat = X_sub[feat_cols].fillna(0).values.astype(np.float32)

    all_preds = {name: [] for name in ["lgb", "xgb", "cb"] if weights.get(name, 0) > 0 or True}

    for seed in seeds:
        if weights.get("lgb", 0) > 0 or len(weights) == 3:
            m = lgb.LGBMClassifier(**{**lgb_params, "random_state": seed})
            m.fit(X_train, y_train)
            all_preds.setdefault("lgb", []).append(m.predict_proba(X_mat)[:, 1])

        if weights.get("xgb", 0) > 0 or len(weights) == 3:
            m = xgb.XGBClassifier(**{**xgb_params, "random_state": seed})
            m.fit(X_train, y_train)
            all_preds.setdefault("xgb", []).append(m.predict_proba(X_mat)[:, 1])

        if weights.get("cb", 0) > 0 or len(weights) == 3:
            if cb_regression:
                cb_reg_p = {k: v for k, v in cb_params.items() if k not in ("loss_function", "eval_metric")}
                cb_reg_p["loss_function"] = "RMSE"
                m = CatBoostRegressor(**{**cb_reg_p, "random_seed": seed})
                m.fit(X_train, y_train.astype(float))
                all_preds.setdefault("cb", []).append(m.predict(X_mat))
            else:
                m = CatBoostClassifier(**{**cb_params, "random_seed": seed})
                m.fit(X_train, y_train)
                all_preds.setdefault("cb", []).append(m.predict_proba(X_mat)[:, 1])

        if weights.get("lr", 0) > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_sub_s = scaler.transform(X_mat)
            lr_m = LogisticRegression(C=lr_C, max_iter=1000, solver="lbfgs")
            lr_m.fit(X_tr_s, y_train)
            all_preds.setdefault("lr", []).append(lr_m.predict_proba(X_sub_s)[:, 1])

        if weights.get("ridge", 0) > 0:
            from sklearn.preprocessing import StandardScaler as SS3
            from sklearn.linear_model import Ridge as RidgeReg3
            scaler3 = SS3()
            X_tr_s3 = scaler3.fit_transform(X_train)
            X_sub_s3 = scaler3.transform(X_mat)
            ridge_sub = RidgeReg3(alpha=1.0)
            ridge_sub.fit(X_tr_s3, y_train.astype(float))
            all_preds.setdefault("ridge", []).append(ridge_sub.predict(X_sub_s3))

        if weights.get("enet", 0) > 0:
            from sklearn.preprocessing import StandardScaler as SS5
            from sklearn.linear_model import SGDClassifier as SGD5
            scaler5 = SS5()
            X_tr_s5 = scaler5.fit_transform(X_train)
            X_sub_s5 = scaler5.transform(X_mat)
            enet_sub = SGD5(
                loss="log_loss", penalty="elasticnet",
                alpha=0.0001, l1_ratio=0.5,
                max_iter=1000, random_state=seed,
            )
            enet_sub.fit(X_tr_s5, y_train)
            all_preds.setdefault("enet", []).append(enet_sub.predict_proba(X_sub_s5)[:, 1])

        if weights.get("et", 0) > 0:
            from sklearn.ensemble import ExtraTreesClassifier
            et_sub = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5,
                                           random_state=seed, n_jobs=-1)
            et_sub.fit(X_train, y_train)
            all_preds.setdefault("et", []).append(et_sub.predict_proba(X_mat)[:, 1])

        if weights.get("knn", 0) > 0:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler as SS6
            scaler6 = SS6()
            X_tr_s6 = scaler6.fit_transform(X_train)
            X_sub_s6 = scaler6.transform(X_mat)
            knn_sub = KNeighborsClassifier(n_neighbors=50, weights="distance")
            knn_sub.fit(X_tr_s6, y_train)
            all_preds.setdefault("knn", []).append(knn_sub.predict_proba(X_sub_s6)[:, 1])

        if weights.get("mlp", 0) > 0:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler as SS8
            scaler8 = SS8()
            X_tr_s8 = scaler8.fit_transform(X_train)
            X_sub_s8 = scaler8.transform(X_mat)
            mlp_sub = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                     early_stopping=True, validation_fraction=0.15,
                                     alpha=0.001, learning_rate="adaptive", random_state=seed)
            mlp_sub.fit(X_tr_s8, y_train)
            all_preds.setdefault("mlp", []).append(mlp_sub.predict_proba(X_sub_s8)[:, 1])

        print(f"    Seed {seed} done")

    intercept = weights.get("_intercept", 0)
    ensemble = np.full(len(X_mat), float(intercept))
    for name in ["lgb", "xgb", "cb", "lr", "ridge", "enet", "et", "knn", "mlp"]:
        w = weights.get(name, 0)
        if w != 0 and name in all_preds and all_preds[name]:
            ensemble += w * np.mean(all_preds[name], axis=0)

    ensemble = np.clip(ensemble, clip_lo, clip_hi)

    # Apply isotonic calibration if provided
    if iso_calibration_data is not None:
        ens_oof, y_oof = iso_calibration_data
        iso = IsotonicRegression(y_min=clip_lo, y_max=clip_hi, out_of_bounds='clip')
        iso.fit(ens_oof, y_oof)
        ensemble = iso.predict(ensemble)

    result = sample_sub.copy()
    result["Pred"] = ensemble
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main(config_path: str):
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    exp_id = cfg["id"]
    exp_dir = EXP_DIR / exp_id
    exp_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = TeeLogger(exp_dir / "log.txt")
    sys.stdout = logger

    t0 = time.time()
    clip_lo, clip_hi = cfg.get("clip", [0.02, 0.98])
    n_seeds = cfg.get("seeds_count", 10)
    eval_seasons = cfg.get("eval_seasons", 10)
    feat_cfg = cfg.get("features", {})

    print("=" * 60)
    print(f"EXPERIMENT: {exp_id}")
    print(f"Hypothesis: {cfg.get('hypothesis', 'N/A')}")
    print(f"Parent: {cfg.get('parent', 'none')}")
    print("=" * 60)

    # ── Stage 1: Load Kaggle Data ──────────────────────────────
    print("\n[1/10] Loading Kaggle data...")
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

    name_to_id = dict(zip(m_teams["TeamName"], m_teams["TeamID"]))
    print(f"  Men: {len(m_reg):,} reg + {len(m_tourney):,} tourney | Women: {len(w_reg):,} reg + {len(w_tourney):,} tourney")

    # ── Stage 2: Load External Data ────────────────────────────
    print("\n[2/10] Loading external data...")
    ext_dfs_m = {}  # Men's external data
    ext_dfs_w = {}  # Women's (currently empty, same schema)

    if feat_cfg.get("kenpom"):
        kp = load_kenpom_efficiency(EXT_DIR, name_to_id)
        if len(kp) > 0:
            ext_dfs_m["kenpom"] = kp.rename(columns={c: f"KP_{c}" for c in kp.columns if c not in ("Season", "TeamID")})
            print(f"  KenPom individual: {len(kp)} rows")

    if feat_cfg.get("barttorvik"):
        bt = load_barttorvik_individual(EXT_DIR, name_to_id)
        if len(bt) > 0:
            bt = bt.rename(columns={"YEAR": "Season"})
            bt = bt.rename(columns={c: f"BT_{c}" for c in bt.columns if c not in ("Season", "TeamID")})
            ext_dfs_m["barttorvik"] = bt
            print(f"  Barttorvik individual: {len(bt)} rows")

    if feat_cfg.get("kp_bt_combined"):
        kpbt = load_kp_bt_combined(EXT_DIR)
        if len(kpbt) > 0:
            ext_dfs_m["kp_bt_combined"] = kpbt
            print(f"  KP+BT combined: {len(kpbt)} rows, {len(kpbt.columns)-2} features")

    if feat_cfg.get("resumes"):
        res = load_resumes(EXT_DIR)
        if len(res) > 0:
            ext_dfs_m["resumes"] = res
            print(f"  Resumes: {len(res)} rows, {len(res.columns)-2} features")

    if feat_cfg.get("evan_miya"):
        em = load_evan_miya(EXT_DIR)
        if len(em) > 0:
            ext_dfs_m["evan_miya"] = em
            print(f"  EvanMiya: {len(em)} rows, {len(em.columns)-2} features")

    if feat_cfg.get("f538"):
        f5 = load_538_ratings(EXT_DIR)
        if len(f5) > 0:
            ext_dfs_m["f538"] = f5
            print(f"  538 Ratings: {len(f5)} rows")

    if feat_cfg.get("coach"):
        try:
            coaches_df = pd.read_csv(DATA_DIR / "MTeamCoaches.csv")
            coach = load_coach_features(EXT_DIR, coaches_df)
            if len(coach) > 0:
                ext_dfs_m["coach"] = coach
                print(f"  Coach features: {len(coach)} rows")
        except Exception as e:
            print(f"  Coach loading failed: {e}")

    if feat_cfg.get("shooting_splits"):
        try:
            ss = pd.read_csv(EXT_DIR / "multisource" / "Shooting Splits.csv")
            ss = ss.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
            # Select key shooting features (offense + defense)
            ss_cols = ["Season", "TeamID"]
            for col in ["THREES FG%", "THREES SHARE", "THREES FG%D", "DUNKS SHARE", "CLOSE TWOS FG%", "CLOSE TWOS SHARE"]:
                if col in ss.columns:
                    clean_name = "SS_" + col.replace(" ", "_").replace("%", "Pct")
                    ss = ss.rename(columns={col: clean_name})
                    ss_cols.append(clean_name)
            ss = ss[ss_cols].dropna()
            ext_dfs_m["shooting_splits"] = ss
            print(f"  Shooting Splits: {len(ss)} rows, {len(ss_cols)-2} features")
        except Exception as e:
            print(f"  Shooting Splits loading failed: {e}")

    if feat_cfg.get("ap_poll"):
        try:
            ap = pd.read_csv(EXT_DIR / "multisource" / "AP Poll Data.csv")
            ap = ap.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
            # Get final week ranking and votes for each team-season
            last_week = ap.groupby(["Season", "TeamID"])["WEEK"].max().reset_index()
            ap_final = ap.merge(last_week, on=["Season", "TeamID", "WEEK"])
            ap_final = ap_final[["Season", "TeamID", "AP VOTES", "AP RANK"]].copy()
            ap_final = ap_final.rename(columns={"AP VOTES": "AP_FinalVotes", "AP RANK": "AP_FinalRank"})
            # Compute momentum: compare last week rank to 4 weeks earlier
            ap_sorted = ap.sort_values(["Season", "TeamID", "WEEK"])
            def compute_momentum(group):
                if len(group) < 5:
                    return pd.Series({"AP_Momentum": 0})
                recent = group.iloc[-1]["AP RANK"]
                earlier = group.iloc[-5]["AP RANK"] if len(group) >= 5 else group.iloc[0]["AP RANK"]
                # Negative = improving (rank going down = better)
                return pd.Series({"AP_Momentum": earlier - recent})
            ap_mom = ap_sorted.groupby(["Season", "TeamID"]).apply(compute_momentum, include_groups=False).reset_index()
            ap_final = ap_final.merge(ap_mom, on=["Season", "TeamID"], how="left")
            ap_final["AP_Momentum"] = ap_final["AP_Momentum"].fillna(0)
            ext_dfs_m["ap_poll"] = ap_final
            print(f"  AP Poll: {len(ap_final)} rows (final rank + momentum)")
        except Exception as e:
            print(f"  AP Poll loading failed: {e}")

    if feat_cfg.get("kenpom_height"):
        try:
            kph = pd.read_csv(EXT_DIR / "kenpom" / "INT _ KenPom _ Height.csv")
            # Map TeamName to TeamID
            kph["TeamID"] = kph["TeamName"].apply(lambda x: name_to_id.get(normalize_name(x)))
            kph = kph.dropna(subset=["TeamID"])
            kph["TeamID"] = kph["TeamID"].astype(int)
            # Select height features — configurable subset
            kph_select = feat_cfg.get("kenpom_height_cols", ["AvgHeight", "EffectiveHeight", "CenterHeight", "Experience", "Bench"])
            all_mappings = {"AvgHeight": "KPH_AvgHgt", "EffectiveHeight": "KPH_EffHgt",
                            "CenterHeight": "KPH_CHgt", "Experience": "KPH_Exp", "Bench": "KPH_Bench"}
            height_cols = ["Season", "TeamID"]
            for col in kph_select:
                new_name = all_mappings.get(col, f"KPH_{col}")
                if col in kph.columns:
                    kph = kph.rename(columns={col: new_name})
                    height_cols.append(new_name)
            kph = kph[height_cols].dropna()
            ext_dfs_m["kenpom_height"] = kph
            print(f"  KenPom Height: {len(kph)} rows, {len(height_cols)-2} features ({kph_select})")
        except Exception as e:
            print(f"  KenPom Height loading failed: {e}")

    if feat_cfg.get("kenpom_point_dist"):
        try:
            kpd = pd.read_csv(EXT_DIR / "kenpom" / "INT _ KenPom _ Point Distribution.csv")
            kpd["TeamID"] = kpd["TeamName"].apply(lambda x: name_to_id.get(normalize_name(x)))
            kpd = kpd.dropna(subset=["TeamID"])
            kpd["TeamID"] = kpd["TeamID"].astype(int)
            pd_cols = ["Season", "TeamID"]
            for col, new_name in [("Off3PtFG", "KPD_Off3Pct"), ("OffFT", "KPD_OffFTPct"),
                                   ("Def3PtFG", "KPD_Def3Pct")]:
                if col in kpd.columns:
                    kpd = kpd.rename(columns={col: new_name})
                    pd_cols.append(new_name)
            kpd = kpd[pd_cols].dropna()
            ext_dfs_m["kenpom_point_dist"] = kpd
            print(f"  KenPom Point Distribution: {len(kpd)} rows, {len(pd_cols)-2} features")
        except Exception as e:
            print(f"  KenPom Point Distribution loading failed: {e}")

    if feat_cfg.get("coach_tourney"):
        try:
            coach_r = pd.read_csv(EXT_DIR / "multisource" / "Coach Results.csv")
            coach_r = coach_r.rename(columns={"YEAR": "Season", "TEAM NO": "TeamID"})
            ct_cols = ["Season", "TeamID"]
            for col, new_name in [("F4%", "Coach_F4Pct"), ("WIN%", "Coach_WinPct"), ("GAMES", "Coach_Games")]:
                if col in coach_r.columns:
                    coach_r = coach_r.rename(columns={col: new_name})
                    ct_cols.append(new_name)
            coach_r = coach_r[ct_cols].dropna()
            ext_dfs_m["coach_tourney"] = coach_r
            print(f"  Coach Tournament: {len(coach_r)} rows, {len(ct_cols)-2} features")
        except Exception as e:
            print(f"  Coach Tournament loading failed: {e}")

    if not ext_dfs_m:
        print("  No external data loaded")

    # Conference tournament features
    m_conf_tourney, w_conf_tourney = None, None
    if feat_cfg.get("conf_tourney", False):
        mct_path = DATA_DIR / "MConferenceTourneyGames.csv"
        wct_path = DATA_DIR / "WConferenceTourneyGames.csv"
        if mct_path.exists():
            m_conf_tourney = compute_conf_tourney(pd.read_csv(mct_path))
            print(f"  Conf tourney Men: {len(m_conf_tourney)} team-seasons")
        if wct_path.exists():
            w_conf_tourney = compute_conf_tourney(pd.read_csv(wct_path))
            print(f"  Conf tourney Women: {len(w_conf_tourney)} team-seasons")

    # ── Stage 3: Elo ───────────────────────────────────────────
    elo_cfg = cfg.get("elo", {})
    if elo_cfg.get("grid_search"):
        print("\n[3/10] Elo grid search...")
        m_seeds_parsed = m_seeds.copy()
        m_seeds_parsed["SeedNum"] = m_seeds_parsed["Seed"].apply(lambda s: int(s[1:3]))
        best_brier, best_params = 1.0, {"k": 32, "home_adv": 100, "carryover": 0.65}
        k_vals = elo_cfg.get("k_values", [24, 32, 40, 48])
        ha_vals = elo_cfg.get("home_adv_values", [64, 82, 100])
        co_vals = elo_cfg.get("carryover_values", [0.55, 0.65, 0.75])
        print(f"  Grid: K={k_vals}, home={ha_vals}, carry={co_vals} ({len(k_vals)*len(ha_vals)*len(co_vals)} combos)")
        for k in k_vals:
            for ha in ha_vals:
                for co in co_vals:
                    elo = compute_elo(m_reg, m_conf, k=k, home_adv=ha, carryover=co)
                    score = quick_elo_eval(elo, m_seeds_parsed, m_tourney, clip_lo, clip_hi)
                    if score < best_brier:
                        best_brier = score
                        best_params = {"k": k, "home_adv": ha, "carryover": co}
        print(f"  Best: K={best_params['k']}, home={best_params['home_adv']}, carry={best_params['carryover']} (Brier={best_brier:.5f})")
    else:
        best_params = {"k": elo_cfg.get("k", 40), "home_adv": elo_cfg.get("home_adv", 64), "carryover": elo_cfg.get("carryover", 0.65)}
        print(f"\n[3/10] Elo (fixed): K={best_params['k']}, home={best_params['home_adv']}, carry={best_params['carryover']}")

    m_elo = compute_elo(m_reg, m_conf, **best_params)
    w_elo = compute_elo(w_reg, w_conf, **best_params)

    # Bradley-Terry (optional)
    m_bt, w_bt = None, None
    if feat_cfg.get("bradley_terry", False):
        print("  Computing Bradley-Terry strengths...")
        m_bt = compute_bt_strengths(m_reg, carryover=best_params["carryover"])
        w_bt = compute_bt_strengths(w_reg, carryover=best_params["carryover"])
        print(f"  BT: {len(m_bt)} M ratings, {len(w_bt)} W ratings")

    # ── Stage 4: Efficiency ────────────────────────────────────
    print("\n[4/10] Computing efficiency...")
    momentum_n = feat_cfg.get("momentum_last_n", 0)
    eff_kwargs = dict(
        last_n=momentum_n,
        consistency=feat_cfg.get("consistency_features", False),
        close_games=feat_cfg.get("close_game_features", False),
        neutral_court=feat_cfg.get("neutral_court_features", False),
        ast_to=feat_cfg.get("ast_to_features", False),
        eff_ratio=feat_cfg.get("eff_ratio", False),
        overtime=feat_cfg.get("overtime_features", False),
    )
    m_eff = compute_efficiency(m_reg_det, **eff_kwargs) if feat_cfg.get("efficiency", True) else pd.DataFrame()
    w_eff = compute_efficiency(w_reg_det, **eff_kwargs) if feat_cfg.get("efficiency", True) else pd.DataFrame()
    if momentum_n > 0:
        print(f"  Momentum: last {momentum_n} games enabled")
    print(f"  Men: {m_eff.shape}, Women: {w_eff.shape}")

    # ── Stage 5: Seeds + Massey ────────────────────────────────
    print("\n[5/10] Seeds + Massey...")
    m_seeds["SeedNum"] = m_seeds["Seed"].apply(lambda s: int(s[1:3]))
    w_seeds["SeedNum"] = w_seeds["Seed"].apply(lambda s: int(s[1:3]))
    m_massey_agg = agg_massey(m_massey) if feat_cfg.get("massey", True) else None
    if m_massey_agg is not None:
        print(f"  Massey Men: {m_massey_agg.shape}")

    # ── Stage 6: Build Team Features ───────────────────────────
    print("\n[6/10] Building team features...")
    if m_conf_tourney is not None:
        ext_dfs_m["conf_tourney"] = m_conf_tourney
    if w_conf_tourney is not None:
        ext_dfs_w["conf_tourney"] = w_conf_tourney
    # Rest days and travel distance (optional)
    m_rest, w_rest, m_travel, w_travel = None, None, None, None
    if feat_cfg.get("rest_days", False):
        m_rest = compute_rest_days(m_reg, m_tourney)
        w_rest = compute_rest_days(w_reg, w_tourney)
        print(f"  RestDays: {len(m_rest)} M + {len(w_rest)} W team-seasons")
    if feat_cfg.get("travel_distance", False):
        # Try Tournament Locations.csv first (best quality, precalculated)
        tourney_loc_path = Path("C:/Users/Admin/Desktop/march_data_temp/multisource/Tournament Locations.csv")
        if tourney_loc_path.exists():
            tloc = pd.read_csv(tourney_loc_path)
            # Use first-round distance (ROUND=64) as team's travel
            first_rd = tloc[tloc.ROUND == 64][["YEAR", "TEAM NO", "DISTANCE (KM)", "TIME ZONES CROSSED"]].copy()
            first_rd.columns = ["Season", "TeamID", "TravelDist_km", "TimezoneCross"]
            m_travel = {(int(r.Season), int(r.TeamID)): r.TravelDist_km
                        for _, r in first_rd.iterrows()}
            w_travel = {}  # Tournament Locations only covers Men's
            print(f"  TravelDist (Tournament Locations): {len(m_travel)} M team-seasons")
        else:
            # Fallback to computed travel data
            travel_path = DATA_DIR / "team_travel_rest.csv"
            if travel_path.exists():
                tdf = pd.read_csv(travel_path)
                tdf_valid = tdf.dropna(subset=["TravelDist_km"])
                m_travel = {(int(r.Season), int(r.TeamID)): r.TravelDist_km
                            for _, r in tdf_valid[tdf_valid.TeamID < 3000].iterrows()}
                w_travel = {(int(r.Season), int(r.TeamID)): r.TravelDist_km
                            for _, r in tdf_valid[tdf_valid.TeamID >= 3000].iterrows()}
                print(f"  TravelDist (computed): {len(m_travel)} M + {len(w_travel)} W team-seasons")
            else:
                print("  WARNING: No travel data found.")

    m_feats = build_team_feats(m_elo, m_eff, m_seeds, m_massey_agg, m_reg, ext_dfs=ext_dfs_m,
                               bt_dict=m_bt, rest_dict=m_rest, travel_dict=m_travel)
    w_feats = build_team_feats(w_elo, w_eff, w_seeds, None, w_reg, ext_dfs=ext_dfs_w,
                               bt_dict=w_bt, rest_dict=w_rest, travel_dict=w_travel)

    # Conference strength features (optional)
    if feat_cfg.get("conference_strength", False):
        print("  Adding conference strength features...")
        for feats, elo_dict, conf_df, label in [
            (m_feats, m_elo, m_conf, "M"),
            (w_feats, w_elo, w_conf, "W"),
        ]:
            elo_rows = [{"Season": s, "TeamID": t, "Elo": e} for (s, t), e in elo_dict.items()]
            elo_df = pd.DataFrame(elo_rows)
            conf_feat = build_conference_features(conf_df, elo_df)
            # Keep only numeric conference features
            conf_cols = ["Season", "TeamID", "ConfEloMean", "ConfEloStd", "ConfRank", "IsPowerConf"]
            conf_feat = conf_feat[conf_cols].drop_duplicates(subset=["Season", "TeamID"])
            before = feats.shape[1]
            merged = feats.merge(conf_feat, on=["Season", "TeamID"], how="left")
            for c in ["ConfEloMean", "ConfEloStd", "ConfRank", "IsPowerConf"]:
                if c in merged.columns:
                    merged[c] = merged[c].fillna(0)
            if label == "M":
                m_feats = merged
            else:
                w_feats = merged
            print(f"    {label}: {before} -> {merged.shape[1]} columns")

    print(f"  Men: {m_feats.shape}, Women: {w_feats.shape}")

    # ── Stage 7: Build Matchup Data ────────────────────────────
    print("\n[7/10] Building matchup data...")
    exclude = {"Season", "TeamID", "Games"}
    m_feat_cols = [c for c in m_feats.columns if c not in exclude and m_feats[c].dtype in [np.float64, np.int64, np.float32]]
    w_feat_cols = [c for c in w_feats.columns if c not in exclude and w_feats[c].dtype in [np.float64, np.int64, np.float32]]

    m_train = make_matchup(m_feats, m_tourney, m_feat_cols)
    m_train["is_mens"] = 1
    w_train = make_matchup(w_feats, w_tourney, w_feat_cols)
    w_train["is_mens"] = 0

    all_cols = sorted(set(m_train.columns) | set(w_train.columns))
    for c in all_cols:
        if c not in m_train.columns:
            m_train[c] = 0
        if c not in w_train.columns:
            w_train[c] = 0

    train = pd.concat([m_train[all_cols], w_train[all_cols]], ignore_index=True).fillna(0)

    # ── Seed Historical Win Rate (matchup-level, leak-free) ──
    if feat_cfg.get("seed_hist_wr", False):
        print("  Adding historical seed matchup win rates (leak-free)...")
        all_seeds_df = pd.concat([m_seeds, w_seeds], ignore_index=True)
        all_tourney_df = pd.concat([m_tourney, w_tourney], ignore_index=True)
        seed_map = all_seeds_df.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

        # Build seed-matchup game log: (Season, StrongSeed, WeakSeed, StrongWon)
        hs = all_seeds_df[["Season", "TeamID", "SeedNum"]]
        ht = all_tourney_df.merge(
            hs.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
            on=["Season", "WTeamID"], how="inner"
        ).merge(
            hs.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
            on=["Season", "LTeamID"], how="inner"
        )
        ht["StrongSeed"] = ht[["WSeed", "LSeed"]].min(axis=1)
        ht["WeakSeed"] = ht[["WSeed", "LSeed"]].max(axis=1)
        ht["StrongWon"] = (ht["WSeed"] == ht["StrongSeed"]).astype(int)

        # Bayesian smoothing parameter (0 = raw rates, >0 = shrink toward global prior)
        bayes_k = feat_cfg.get("seed_hist_wr_bayes_k", 0)

        # Precompute cumulative rates per season (leak-free: only prior seasons)
        unique_seasons = sorted(train["Season"].unique())
        season_rates = {}  # season -> {(strong, weak): (rate, count)}
        for s in unique_seasons:
            hist = ht[ht.Season < s]
            if len(hist) == 0:
                season_rates[s] = {}
                continue
            if bayes_k > 0:
                # Store (mean, count) for Bayesian smoothing
                grouped = hist.groupby(["StrongSeed", "WeakSeed"])["StrongWon"]
                rates = {k: (v.mean(), v.count()) for k, v in grouped}
                season_rates[s] = rates
            else:
                rates = hist.groupby(["StrongSeed", "WeakSeed"])["StrongWon"].mean().to_dict()
                season_rates[s] = rates

        # Also compute rates using ALL data (for submissions)
        if bayes_k > 0:
            grouped_all = ht.groupby(["StrongSeed", "WeakSeed"])["StrongWon"]
            all_rates = {k: (v.mean(), v.count()) for k, v in grouped_all}
        else:
            all_rates = ht.groupby(["StrongSeed", "WeakSeed"])["StrongWon"].mean().to_dict()

        # Global strong-seed win rate (prior for Bayesian smoothing)
        global_prior = ht["StrongWon"].mean() if len(ht) > 0 else 0.5

        def _get_seed_wr(s1, s2, rates_dict):
            strong, weak = min(s1, s2), max(s1, s2)
            if strong == weak:
                return 0.5
            val = rates_dict.get((strong, weak), None)
            if val is None:
                return global_prior if s1 <= s2 else 1.0 - global_prior
            if bayes_k > 0 and isinstance(val, tuple):
                observed, n = val
                rate = (n * observed + bayes_k * global_prior) / (n + bayes_k)
            else:
                rate = val
            return rate if s1 <= s2 else 1.0 - rate

        seed_wr_vals = []
        for _, row in train.iterrows():
            s = int(row["Season"])
            t1, t2 = int(row["TeamID1"]), int(row["TeamID2"])
            s1 = seed_map.get((s, t1), 8)
            s2 = seed_map.get((s, t2), 8)
            seed_wr_vals.append(_get_seed_wr(s1, s2, season_rates.get(s, {})))
        train["SeedHistWR"] = seed_wr_vals
        print(f"  SeedHistWR: mean={np.mean(seed_wr_vals):.4f}, std={np.std(seed_wr_vals):.4f}")

        # Gender-specific seed win rates (optional)
        if feat_cfg.get("seed_hist_wr_by_gender", False):
            print("  Adding gender-specific SeedHistWR...")
            # Build separate M/W game logs
            for gender_label, t_df, s_df in [("M", m_tourney, m_seeds), ("W", w_tourney, w_seeds)]:
                gs = s_df[["Season", "TeamID", "SeedNum"]]
                gt = t_df.merge(
                    gs.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}),
                    on=["Season", "WTeamID"], how="inner"
                ).merge(
                    gs.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}),
                    on=["Season", "LTeamID"], how="inner"
                )
                gt["StrongSeed"] = gt[["WSeed", "LSeed"]].min(axis=1)
                gt["WeakSeed"] = gt[["WSeed", "LSeed"]].max(axis=1)
                gt["StrongWon"] = (gt["WSeed"] == gt["StrongSeed"]).astype(int)

                g_season_rates = {}
                for s in unique_seasons:
                    gh = gt[gt.Season < s]
                    g_season_rates[s] = gh.groupby(["StrongSeed", "WeakSeed"])["StrongWon"].mean().to_dict() if len(gh) > 0 else {}

                col_name = f"SeedHistWR_{gender_label}"
                g_vals = []
                is_gender = 1 if gender_label == "M" else 0
                for _, row in train.iterrows():
                    if int(row.get("is_mens", 1)) == is_gender:
                        s = int(row["Season"])
                        t1, t2 = int(row["TeamID1"]), int(row["TeamID2"])
                        s1 = seed_map.get((s, t1), 8)
                        s2 = seed_map.get((s, t2), 8)
                        g_vals.append(_get_seed_wr(s1, s2, g_season_rates.get(s, {})))
                    else:
                        g_vals.append(0.5)  # neutral for other gender
                train[col_name] = g_vals
                print(f"    {col_name}: mean={np.mean(g_vals):.4f}")

            # Store gender-specific all_rates for submissions
            # Recompute M rates from scratch (loop variable gt is W from last iteration)
            m_gs = m_seeds[["Season", "TeamID", "SeedNum"]]
            m_gt = m_tourney.merge(
                m_gs.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}), on=["Season", "WTeamID"], how="inner"
            ).merge(
                m_gs.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}), on=["Season", "LTeamID"], how="inner"
            )
            m_gt["StrongSeed"] = m_gt[["WSeed", "LSeed"]].min(axis=1)
            m_gt["WeakSeed"] = m_gt[["WSeed", "LSeed"]].max(axis=1)
            m_gt["StrongWon"] = (m_gt["WSeed"] == m_gt["StrongSeed"]).astype(int)
            all_rates_m = m_gt.groupby(["StrongSeed", "WeakSeed"])["StrongWon"].mean().to_dict()
            # W rates from loop variable gt (last iteration was W)
            w_gs = w_seeds[["Season", "TeamID", "SeedNum"]]
            w_gt = w_tourney.merge(
                w_gs.rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}), on=["Season", "WTeamID"], how="inner"
            ).merge(
                w_gs.rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}), on=["Season", "LTeamID"], how="inner"
            )
            w_gt["StrongSeed"] = w_gt[["WSeed", "LSeed"]].min(axis=1)
            w_gt["WeakSeed"] = w_gt[["WSeed", "LSeed"]].max(axis=1)
            w_gt["StrongWon"] = (w_gt["WSeed"] == w_gt["StrongSeed"]).astype(int)
            all_rates_w = w_gt.groupby(["StrongSeed", "WeakSeed"])["StrongWon"].mean().to_dict()

    # ── Interaction Features (optional) ──────────────────────
    if feat_cfg.get("interaction_features", False):
        print("  Adding interaction features...")
        # EffRatio: offensive efficiency / defensive efficiency (team-level ratio as diff)
        if "OffEff_diff" in train.columns and "DefEff_diff" in train.columns:
            # Use raw team values to compute ratio, then diff is already captured
            # Instead: create a combined efficiency ratio feature
            train["EffRatio_diff"] = train["OffEff_diff"] - train["DefEff_diff"]  # already NetEff, skip
        # Elo * SeedHistWR interaction
        if "Elo_diff" in train.columns and "SeedHistWR" in train.columns:
            train["Elo_x_SeedHist"] = train["Elo_diff"] * train["SeedHistWR"]
            print(f"    Elo_x_SeedHist: mean={train['Elo_x_SeedHist'].mean():.2f}")
        # Elo * SeedNum interaction
        if "Elo_diff" in train.columns and "SeedNum_diff" in train.columns:
            train["Elo_x_Seed"] = train["Elo_diff"] * train["SeedNum_diff"]
            print(f"    Elo_x_Seed: mean={train['Elo_x_Seed'].mean():.2f}")
        # SeedHistWR * is_mens (gender-specific seed dynamics)
        if "SeedHistWR" in train.columns and "is_mens" in train.columns:
            train["SeedHist_x_Gender"] = train["SeedHistWR"] * train["is_mens"]
            print(f"    SeedHist_x_Gender: mean={train['SeedHist_x_Gender'].mean():.4f}")
        # Drop EffRatio_diff if it's just NetEff
        if "EffRatio_diff" in train.columns:
            train = train.drop(columns=["EffRatio_diff"])

    meta = ["Season", "TeamID1", "TeamID2", "target"]
    feat_cols = sorted([c for c in train.columns if c not in meta])

    # ── Feature Selection (optional) ──────────────────────────
    feat_sel = cfg.get("feature_selection", {})
    top_n = feat_sel.get("top_n", 0)
    if top_n > 0 and top_n < len(feat_cols):
        print(f"\n  Feature selection: picking top {top_n} from {len(feat_cols)}...")
        X_all = train[feat_cols].fillna(0).values.astype(np.float32)
        y_all = train["target"].values
        selector = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05,
            max_depth=6, verbosity=-1, n_jobs=-1, random_state=42,
        )
        selector.fit(X_all, y_all)
        imp = sorted(zip(feat_cols, selector.feature_importances_), key=lambda x: -x[1])
        feat_cols = [name for name, _ in imp[:top_n]]
        print(f"  Selected: {feat_cols}")

    X = train[feat_cols].values.astype(np.float32)
    y = train["target"].values
    seasons = train["Season"].values

    print(f"  Training: {train.shape}, Features ({len(feat_cols)}): {feat_cols[:10]}...")

    # ── Stage 8: Optuna CatBoost Tuning ────────────────────────
    models_cfg = cfg.get("models", {})
    cb_optuna_trials = models_cfg.get("cb_optuna_trials", 0)
    cb_reuse = models_cfg.get("cb_reuse_params")
    all_seasons = sorted(np.unique(seasons))
    eval_szns = all_seasons[-eval_seasons:]

    if cb_reuse:
        # Load params from parent experiment
        parent_results_path = EXP_DIR / cb_reuse / "results.json"
        if parent_results_path.exists():
            with open(parent_results_path) as f:
                parent_results = json.load(f)
            best_cb_params = parent_results.get("cb_optuna_params", {})
            best_cb_params["loss_function"] = "Logloss"
            best_cb_params["eval_metric"] = "Logloss"
            best_cb_params["verbose"] = 0
            print(f"\n[8/10] Reusing CatBoost params from {cb_reuse}")
            # Apply any overrides from config
            cb_overrides = models_cfg.get("cb_params_override", {})
            if cb_overrides:
                best_cb_params.update(cb_overrides)
                print(f"  CB param overrides: {cb_overrides}")
        else:
            print(f"\n[8/10] Parent {cb_reuse} not found, using defaults")
            best_cb_params = {"loss_function": "Logloss", "eval_metric": "Logloss", "verbose": 0,
                              "learning_rate": 0.03, "depth": 5, "iterations": 400}
    elif cb_optuna_trials > 0:
        print(f"\n[8/10] Optuna CatBoost tuning ({cb_optuna_trials} trials)...")

        def objective(trial):
            params = {
                "loss_function": "Logloss", "eval_metric": "Logloss",
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
                tr_idx, va_idx = seasons < vs, seasons == vs
                if va_idx.sum() == 0:
                    continue
                m = CatBoostClassifier(**{**params, "random_seed": 42})
                m.fit(X[tr_idx], y[tr_idx])
                pred = np.clip(m.predict_proba(X[va_idx])[:, 1], clip_lo, clip_hi)
                scores.append(brier_score_loss(y[va_idx], pred))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=cb_optuna_trials)
        best_cb_params = study.best_params
        best_cb_params["loss_function"] = "Logloss"
        best_cb_params["eval_metric"] = "Logloss"
        best_cb_params["verbose"] = 0
        print(f"  Best Brier: {study.best_value:.5f}")
    else:
        print(f"\n[8/10] CatBoost with default params")
        best_cb_params = {"loss_function": "Logloss", "eval_metric": "Logloss", "verbose": 0,
                          "learning_rate": 0.03, "depth": 5, "iterations": 400}

    # ── Stage 9: Temporal CV ───────────────────────────────────
    separate_mw = feat_cfg.get("separate_models_mw", False)
    if separate_mw:
        print(f"\n[9/10] Temporal CV — SEPARATE M/W models (Brier, last {eval_seasons} seasons)...")
    else:
        print(f"\n[9/10] Temporal CV (Brier, last {eval_seasons} seasons)...")

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

    # For separate M/W: identify is_mens column index and build gender mask
    if separate_mw and "is_mens" in feat_cols:
        is_mens_idx = feat_cols.index("is_mens")
        gender_mask = X[:, is_mens_idx] > 0.5  # True = men's
        # Use features WITHOUT is_mens for separate models
        mw_feat_idx = [i for i, c in enumerate(feat_cols) if c != "is_mens"]
    elif separate_mw:
        # is_mens not in features, use train data
        gender_mask = train["is_mens"].values > 0.5
        mw_feat_idx = list(range(len(feat_cols)))
    else:
        gender_mask = None
        mw_feat_idx = None

    # ── Season weighting (optional) ──────────────────────────
    season_decay = cfg.get("season_weight_decay", 0)
    if season_decay > 0:
        max_season = seasons.max()
        sample_weights = np.array([season_decay ** (max_season - s) for s in seasons], dtype=np.float64)
        print(f"  Season weighting: decay={season_decay}, weight range=[{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    else:
        sample_weights = None

    lr_C = models_cfg.get("lr_C", 1.0)

    lgb_oof = np.full(len(y), np.nan)
    xgb_oof = np.full(len(y), np.nan)
    cb_oof = np.full(len(y), np.nan)
    lr_oof = np.full(len(y), np.nan)
    ridge_oof = np.full(len(y), np.nan)
    enet_oof = np.full(len(y), np.nan)
    et_oof = np.full(len(y), np.nan)
    knn_oof = np.full(len(y), np.nan)
    mlp_oof = np.full(len(y), np.nan)
    lgb_scores, xgb_scores, cb_scores, lr_scores, ridge_scores, enet_scores = [], [], [], [], [], []
    et_scores, knn_scores, mlp_scores = [], [], []

    cb_regression = models_cfg.get("cb_regression", False)

    def _train_predict(model_cls, params, X_tr, y_tr, X_va, seed_key="random_state", w_tr=None, regression=False):
        m = model_cls(**{**params, seed_key: 42})
        if w_tr is not None:
            m.fit(X_tr, y_tr, sample_weight=w_tr)
        else:
            m.fit(X_tr, y_tr)
        if regression:
            return m.predict(X_va)
        return m.predict_proba(X_va)[:, 1]

    for vs in eval_szns:
        tr_mask, va_mask = seasons < vs, seasons == vs
        if va_mask.sum() == 0:
            continue

        if separate_mw:
            # Train separate models for M and W, combine predictions
            X_mw = X[:, mw_feat_idx]
            for model_name, model_cls, oof_arr, scores_list, params, seed_key, enabled in [
                ("lgb", lgb.LGBMClassifier, lgb_oof, lgb_scores, lgb_params, "random_state", models_cfg.get("lgb", True)),
                ("xgb", xgb.XGBClassifier, xgb_oof, xgb_scores, xgb_params, "random_state", models_cfg.get("xgb", True)),
                ("cb", CatBoostClassifier, cb_oof, cb_scores, best_cb_params, "random_seed", models_cfg.get("cb", True)),
            ]:
                if not enabled:
                    continue
                preds_va = np.full(va_mask.sum(), np.nan)
                for is_m, label in [(True, "M"), (False, "W")]:
                    g = gender_mask if is_m else ~gender_mask
                    g_tr = tr_mask & g
                    g_va = va_mask & g
                    if g_tr.sum() < 10 or g_va.sum() == 0:
                        continue
                    p = _train_predict(model_cls, params, X_mw[g_tr], y[g_tr], X_mw[g_va], seed_key)
                    # Map back to va positions
                    va_indices = np.where(va_mask)[0]
                    g_va_in_va = g[va_mask]
                    preds_va[g_va_in_va] = p
                p_clipped = np.clip(preds_va, clip_lo, clip_hi)
                oof_arr[va_mask] = p_clipped
                valid_preds = ~np.isnan(preds_va)
                if valid_preds.any():
                    scores_list.append(brier_score_loss(y[va_mask][valid_preds], p_clipped[valid_preds]))
        else:
            X_tr, X_va, y_tr, y_va = X[tr_mask], X[va_mask], y[tr_mask], y[va_mask]
            w_tr = sample_weights[tr_mask] if sample_weights is not None else None

            if models_cfg.get("lgb", True):
                p = np.clip(_train_predict(lgb.LGBMClassifier, lgb_params, X_tr, y_tr, X_va, w_tr=w_tr), clip_lo, clip_hi)
                lgb_oof[va_mask] = p
                lgb_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("xgb", True):
                p = np.clip(_train_predict(xgb.XGBClassifier, xgb_params, X_tr, y_tr, X_va, w_tr=w_tr), clip_lo, clip_hi)
                xgb_oof[va_mask] = p
                xgb_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("cb", True):
                if cb_regression:
                    cb_reg_params = {k: v for k, v in best_cb_params.items() if k not in ("loss_function", "eval_metric")}
                    cb_reg_params["loss_function"] = "RMSE"
                    p = np.clip(_train_predict(CatBoostRegressor, cb_reg_params, X_tr, y_tr.astype(float), X_va, "random_seed", w_tr=w_tr, regression=True), clip_lo, clip_hi)
                else:
                    p = np.clip(_train_predict(CatBoostClassifier, best_cb_params, X_tr, y_tr, X_va, "random_seed", w_tr=w_tr), clip_lo, clip_hi)
                cb_oof[va_mask] = p
                cb_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("lr", False):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_va_s = scaler.transform(X_va)
                lr_model = LogisticRegression(C=lr_C, max_iter=1000, solver="lbfgs")
                if w_tr is not None:
                    lr_model.fit(X_tr_s, y_tr, sample_weight=w_tr)
                else:
                    lr_model.fit(X_tr_s, y_tr)
                p = np.clip(lr_model.predict_proba(X_va_s)[:, 1], clip_lo, clip_hi)
                lr_oof[va_mask] = p
                lr_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("ridge", False):
                from sklearn.linear_model import Ridge as RidgeReg
                from sklearn.preprocessing import StandardScaler as SS2
                scaler2 = SS2()
                X_tr_s2 = scaler2.fit_transform(X_tr)
                X_va_s2 = scaler2.transform(X_va)
                ridge_alpha = models_cfg.get("ridge_alpha", 1.0)
                ridge_m = RidgeReg(alpha=ridge_alpha)
                if w_tr is not None:
                    ridge_m.fit(X_tr_s2, y_tr.astype(float), sample_weight=w_tr)
                else:
                    ridge_m.fit(X_tr_s2, y_tr.astype(float))
                p = np.clip(ridge_m.predict(X_va_s2), clip_lo, clip_hi)
                ridge_oof[va_mask] = p
                ridge_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("elasticnet", False):
                from sklearn.linear_model import SGDClassifier
                from sklearn.preprocessing import StandardScaler as SS4
                scaler4 = SS4()
                X_tr_s4 = scaler4.fit_transform(X_tr)
                X_va_s4 = scaler4.transform(X_va)
                enet_alpha = models_cfg.get("enet_alpha", 0.0001)
                enet_l1_ratio = models_cfg.get("enet_l1_ratio", 0.5)
                enet_m = SGDClassifier(
                    loss="log_loss", penalty="elasticnet",
                    alpha=enet_alpha, l1_ratio=enet_l1_ratio,
                    max_iter=1000, random_state=42,
                )
                if w_tr is not None:
                    enet_m.fit(X_tr_s4, y_tr, sample_weight=w_tr)
                else:
                    enet_m.fit(X_tr_s4, y_tr)
                p = np.clip(enet_m.predict_proba(X_va_s4)[:, 1], clip_lo, clip_hi)
                enet_oof[va_mask] = p
                enet_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("extratrees", False):
                from sklearn.ensemble import ExtraTreesClassifier
                et_n = models_cfg.get("et_n_estimators", 500)
                et_model = ExtraTreesClassifier(n_estimators=et_n, max_depth=models_cfg.get("et_max_depth", None),
                                                 min_samples_leaf=models_cfg.get("et_min_samples_leaf", 5),
                                                 random_state=42, n_jobs=-1)
                if w_tr is not None:
                    et_model.fit(X_tr, y_tr, sample_weight=w_tr)
                else:
                    et_model.fit(X_tr, y_tr)
                p = np.clip(et_model.predict_proba(X_va)[:, 1], clip_lo, clip_hi)
                et_oof[va_mask] = p
                et_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("knn", False):
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.preprocessing import StandardScaler as SS5
                scaler5 = SS5()
                X_tr_s5 = scaler5.fit_transform(X_tr)
                X_va_s5 = scaler5.transform(X_va)
                knn_k = models_cfg.get("knn_k", 50)
                knn_model = KNeighborsClassifier(n_neighbors=knn_k, weights="distance", metric="minkowski", p=2)
                knn_model.fit(X_tr_s5, y_tr)
                p = np.clip(knn_model.predict_proba(X_va_s5)[:, 1], clip_lo, clip_hi)
                knn_oof[va_mask] = p
                knn_scores.append(brier_score_loss(y_va, p))

            if models_cfg.get("mlp", False):
                from sklearn.neural_network import MLPClassifier
                from sklearn.preprocessing import StandardScaler as SS7
                scaler7 = SS7()
                X_tr_s7 = scaler7.fit_transform(X_tr)
                X_va_s7 = scaler7.transform(X_va)
                mlp_layers = tuple(models_cfg.get("mlp_layers", [64, 32]))
                mlp_model = MLPClassifier(hidden_layer_sizes=mlp_layers, max_iter=500,
                                           early_stopping=True, validation_fraction=0.15,
                                           alpha=models_cfg.get("mlp_alpha", 0.001),
                                           learning_rate="adaptive", random_state=42)
                mlp_model.fit(X_tr_s7, y_tr)
                p = np.clip(mlp_model.predict_proba(X_va_s7)[:, 1], clip_lo, clip_hi)
                mlp_oof[va_mask] = p
                mlp_scores.append(brier_score_loss(y_va, p))

        lr_str = f" | LR={lr_scores[-1]:.5f}" if lr_scores and len(lr_scores) == len(cb_scores) else ""
        ridge_str = f" | Ridge={ridge_scores[-1]:.5f}" if ridge_scores and len(ridge_scores) == len(cb_scores) else ""
        enet_str = f" | ENet={enet_scores[-1]:.5f}" if enet_scores and len(enet_scores) == len(cb_scores) else ""
        et_str = f" | ET={et_scores[-1]:.5f}" if et_scores and len(et_scores) == len(cb_scores) else ""
        knn_str = f" | KNN={knn_scores[-1]:.5f}" if knn_scores and len(knn_scores) == len(cb_scores) else ""
        mlp_str = f" | MLP={mlp_scores[-1]:.5f}" if mlp_scores and len(mlp_scores) == len(cb_scores) else ""
        print(f"  {int(vs)} | n={va_mask.sum():3d} | LGB={lgb_scores[-1] if lgb_scores else 0:.5f} | XGB={xgb_scores[-1] if xgb_scores else 0:.5f} | CB={cb_scores[-1] if cb_scores else 0:.5f}{lr_str}{ridge_str}{enet_str}{et_str}{knn_str}{mlp_str}")

    mean_lgb = np.mean(lgb_scores) if lgb_scores else 1.0
    mean_xgb = np.mean(xgb_scores) if xgb_scores else 1.0
    mean_cb = np.mean(cb_scores) if cb_scores else 1.0
    mean_lr = np.mean(lr_scores) if lr_scores else 1.0
    mean_ridge = np.mean(ridge_scores) if ridge_scores else 1.0
    mean_enet = np.mean(enet_scores) if enet_scores else 1.0
    mean_et = np.mean(et_scores) if et_scores else 1.0
    mean_knn = np.mean(knn_scores) if knn_scores else 1.0
    mean_mlp = np.mean(mlp_scores) if mlp_scores else 1.0
    lr_str = f" | LR: {mean_lr:.5f}" if lr_scores else ""
    ridge_str = f" | Ridge: {mean_ridge:.5f}" if ridge_scores else ""
    enet_str = f" | ENet: {mean_enet:.5f}" if enet_scores else ""
    et_str = f" | ET: {mean_et:.5f}" if et_scores else ""
    knn_str = f" | KNN: {mean_knn:.5f}" if knn_scores else ""
    mlp_str = f" | MLP: {mean_mlp:.5f}" if mlp_scores else ""
    print(f"\n  LGB: {mean_lgb:.5f} | XGB: {mean_xgb:.5f} | CB: {mean_cb:.5f}{lr_str}{ridge_str}{enet_str}{et_str}{knn_str}{mlp_str}")

    # Ensemble
    valid = np.ones(len(y), dtype=bool)
    preds_list, names = [], []
    if lgb_scores:
        valid &= ~np.isnan(lgb_oof)
        preds_list.append(lgb_oof)
        names.append("lgb")
    if xgb_scores:
        valid &= ~np.isnan(xgb_oof)
        preds_list.append(xgb_oof)
        names.append("xgb")
    if cb_scores:
        valid &= ~np.isnan(cb_oof)
        preds_list.append(cb_oof)
        names.append("cb")
    if lr_scores:
        valid &= ~np.isnan(lr_oof)
        preds_list.append(lr_oof)
        names.append("lr")
    if ridge_scores:
        valid &= ~np.isnan(ridge_oof)
        preds_list.append(ridge_oof)
        names.append("ridge")
    if enet_scores:
        valid &= ~np.isnan(enet_oof)
        preds_list.append(enet_oof)
        names.append("enet")
    if et_scores:
        valid &= ~np.isnan(et_oof)
        preds_list.append(et_oof)
        names.append("et")
    if knn_scores:
        valid &= ~np.isnan(knn_oof)
        preds_list.append(knn_oof)
        names.append("knn")
    if mlp_scores:
        valid &= ~np.isnan(mlp_oof)
        preds_list.append(mlp_oof)
        names.append("mlp")

    ensemble_method = cfg.get("ensemble", {}).get("method", "hill_climbing")

    if ensemble_method == "stacking":
        # Stacking: Ridge regression on OOF predictions
        from sklearn.linear_model import Ridge
        oof_matrix = np.column_stack([p[valid] for p in preds_list])
        y_valid = y[valid]
        seasons_valid = seasons[valid]
        # Use CV to find best alpha
        best_alpha, best_stack_brier = 1.0, 1.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            stack_preds = np.full(len(y_valid), np.nan)
            for vs in eval_szns:
                s_tr = seasons_valid < vs
                s_va = seasons_valid == vs
                if s_va.sum() == 0 or s_tr.sum() == 0:
                    continue
                ridge = Ridge(alpha=alpha)
                ridge.fit(oof_matrix[s_tr], y_valid[s_tr])
                stack_preds[s_va] = ridge.predict(oof_matrix[s_va])
            has_pred = ~np.isnan(stack_preds)
            stack_preds_clipped = np.clip(stack_preds[has_pred], clip_lo, clip_hi)
            sb = brier_score_loss(y_valid[has_pred], stack_preds_clipped)
            if sb < best_stack_brier:
                best_stack_brier = sb
                best_alpha = alpha
        # Refit final stacking model
        stack_model = Ridge(alpha=best_alpha)
        stack_model.fit(oof_matrix, y_valid)
        hc_weights = {n: round(w, 4) for n, w in zip(names, stack_model.coef_)}
        hc_weights["_intercept"] = round(float(stack_model.intercept_), 4)
        hc_brier = best_stack_brier
        print(f"  Stacking Brier: {hc_brier:.5f} | Alpha: {best_alpha} | Coefs: {hc_weights} | Intercept: {stack_model.intercept_:.4f}")
    else:
        hc_weights_list, hc_brier = hill_climbing(
            [p[valid] for p in preds_list], y[valid], clip_lo, clip_hi,
            n_iter=cfg.get("ensemble", {}).get("n_iter", 500),
            step=cfg.get("ensemble", {}).get("step", 0.01),
        )
        hc_weights = {n: round(w, 3) for n, w in zip(names, hc_weights_list)}
        print(f"  HC Brier: {hc_brier:.5f} | Weights: {hc_weights}")

    # ── Post-ensemble isotonic calibration (optional) ──────────
    post_cal = cfg.get("post_ensemble_calibration", False)
    if post_cal:
        print(f"\n  Post-ensemble isotonic calibration...")
        # Compute ensemble OOF predictions using HC weights
        ens_oof = np.zeros(valid.sum())
        for p, n in zip(preds_list, names):
            w = hc_weights.get(n, 0)
            if isinstance(w, (int, float)) and w > 0:
                ens_oof += w * p[valid]
        ens_oof = np.clip(ens_oof, clip_lo, clip_hi)

        # Temporal CV isotonic calibration
        y_valid = y[valid]
        seasons_valid = seasons[valid]
        iso_preds = np.full(len(y_valid), np.nan)
        for vs in eval_szns:
            tr_mask = seasons_valid < vs
            va_mask = seasons_valid == vs
            if va_mask.sum() == 0 or tr_mask.sum() == 0:
                continue
            iso = IsotonicRegression(y_min=clip_lo, y_max=clip_hi, out_of_bounds='clip')
            iso.fit(ens_oof[tr_mask], y_valid[tr_mask])
            iso_preds[va_mask] = iso.predict(ens_oof[va_mask])

        has_pred = ~np.isnan(iso_preds)
        if has_pred.sum() > 0:
            iso_brier = brier_score_loss(y_valid[has_pred], iso_preds[has_pred])
            print(f"  Isotonic Brier: {iso_brier:.5f} (HC was {hc_brier:.5f}, delta={iso_brier - hc_brier:+.5f})")
            if iso_brier < hc_brier:
                hc_brier = iso_brier
                hc_weights["_isotonic"] = True
                print(f"  *** Isotonic calibration IMPROVED! Using calibrated predictions.")

    # Feature importance
    model_full = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    model_full.fit(X, y)
    imp = sorted(zip(feat_cols, model_full.feature_importances_), key=lambda x: -x[1])[:20]
    print(f"\n  Top features:")
    for name_f, val in imp[:10]:
        print(f"    {name_f:40s} {val:5d}")

    # ── Stage 10: Submissions ──────────────────────────────────
    print(f"\n[10/10] Generating submissions...")
    seeds_to_use = SEEDS_LIST[:n_seeds]

    seed_hist = None
    if feat_cfg.get("seed_hist_wr", False):
        # Pre-smooth all_rates for submissions (convert tuples to plain rates)
        bayes_k = feat_cfg.get("seed_hist_wr_bayes_k", 0)
        if bayes_k > 0 and all_rates and isinstance(next(iter(all_rates.values())), tuple):
            gp = ht["StrongWon"].mean() if len(ht) > 0 else 0.5
            sub_rates = {k: (n * obs + bayes_k * gp) / (n + bayes_k) for k, (obs, n) in all_rates.items()}
        else:
            sub_rates = all_rates
        seed_hist = (sub_rates, seed_map)
        if feat_cfg.get("seed_hist_wr_by_gender", False):
            seed_hist = (sub_rates, seed_map, all_rates_m, all_rates_w)

    # Prepare isotonic calibration data for submissions if it improved
    iso_cal_data = None
    if post_cal and hc_weights.get("_isotonic", False):
        # Compute full ensemble OOF for isotonic fit
        ens_oof_full = np.zeros(valid.sum())
        for p, n in zip(preds_list, names):
            w = hc_weights.get(n, 0)
            if isinstance(w, (int, float)) and w > 0:
                ens_oof_full += w * p[valid]
        ens_oof_full = np.clip(ens_oof_full, clip_lo, clip_hi)
        iso_cal_data = (ens_oof_full, y[valid])

    for stage_name, sample_sub in [("stage1", sub1), ("stage2", sub2)]:
        print(f"\n  {stage_name}...")
        result = predict_submission(
            sample_sub, m_feats, w_feats, feat_cols, seeds_to_use, hc_weights,
            X, y, lgb_params, xgb_params, best_cb_params, clip_lo, clip_hi,
            seed_hist_data=seed_hist, lr_C=lr_C, cb_regression=cb_regression,
            iso_calibration_data=iso_cal_data,
        )
        sub_path = SUB_DIR / f"submission_{stage_name}_{exp_id}.csv"
        result.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path} | mean={result.Pred.mean():.4f}, std={result.Pred.std():.4f}")

    elapsed = time.time() - t0

    # ── Save Results ───────────────────────────────────────────
    results = {
        "id": exp_id,
        "status": "completed",
        "elapsed_seconds": round(elapsed),
        "cv_brier": {
            "lgb": round(mean_lgb, 5), "xgb": round(mean_xgb, 5), "cb": round(mean_cb, 5),
            "ensemble_hc": round(hc_brier, 5),
        },
        "hc_weights": hc_weights,
        "elo_params": best_params,
        "cb_optuna_params": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in best_cb_params.items() if k != "verbose"},
        "n_features": len(feat_cols),
        "feature_importance_top20": [[n, int(v)] for n, v in imp],
        "clip_range": [clip_lo, clip_hi],
    }

    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Update registry
    registry_path = EXP_DIR / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"best_cv_brier": 1.0, "best_experiment_id": None, "total_experiments": 0, "experiments": []}

    exp_record = {
        "id": exp_id,
        "name": cfg.get("hypothesis", exp_id)[:80],
        "parent_id": cfg.get("parent"),
        "status": "completed",
        "elapsed_seconds": round(elapsed),
        "cv_brier": round(hc_brier, 5),
        "hc_weights": hc_weights,
        "n_features": len(feat_cols),
        "key_findings": "",
        "tags": [],
    }

    # Remove existing entry if re-running
    registry["experiments"] = [e for e in registry["experiments"] if e["id"] != exp_id]
    registry["experiments"].append(exp_record)
    registry["total_experiments"] = len(registry["experiments"])

    if hc_brier < registry.get("best_cv_brier", 1.0):
        registry["best_cv_brier"] = round(hc_brier, 5)
        registry["best_experiment_id"] = exp_id
        print(f"\n  *** NEW BEST: {hc_brier:.5f} ***")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s | CV Brier: {hc_brier:.5f} | Features: {len(feat_cols)}")
    print(f"{'='*60}")

    # Restore stdout
    sys.stdout = logger.terminal
    logger.close()

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
