"""
Experiment C - Step 1: Bradley-Terry standalone evaluation.
Compare BT strengths vs Elo for predicting tournament outcomes.
If BT standalone Brier < 0.18, proceed with integration into run_experiment.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def compute_bt_strengths(reg_df, conf_df, carryover=0.65, l2_reg=0.01):
    """Compute Bradley-Terry team strengths per season via MLE."""
    strengths = {}
    seasons = sorted(reg_df["Season"].unique())

    for season in seasons:
        sg = reg_df[reg_df.Season == season]
        teams = sorted(set(sg["WTeamID"]) | set(sg["LTeamID"]))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Init from previous season with carryover
        theta0 = np.zeros(n + 1)  # +1 for home advantage
        for i, t in enumerate(teams):
            prev = strengths.get((season - 1, t), 0.0)
            theta0[i] = carryover * prev
        theta0[-1] = 0.3  # home advantage init

        # Build game arrays for vectorized computation
        w_idx = []
        l_idx = []
        home_flags = []  # +1 if winner home, -1 if winner away, 0 if neutral
        for _, g in sg.iterrows():
            wi = team_to_idx.get(int(g.WTeamID))
            li = team_to_idx.get(int(g.LTeamID))
            if wi is not None and li is not None:
                w_idx.append(wi)
                l_idx.append(li)
                if g.WLoc == "H":
                    home_flags.append(1)
                elif g.WLoc == "A":
                    home_flags.append(-1)
                else:
                    home_flags.append(0)

        w_idx = np.array(w_idx)
        l_idx = np.array(l_idx)
        home_flags = np.array(home_flags, dtype=np.float64)

        def neg_log_lik(x):
            theta = x[:n]
            h = x[n]
            diff = theta[w_idx] - theta[l_idx] + h * home_flags
            # -log(sigmoid(diff)) = log(1 + exp(-diff))
            nll = np.sum(np.logaddexp(0, -diff))
            reg = l2_reg * np.sum(theta ** 2)
            return nll + reg

        def grad(x):
            theta = x[:n]
            h = x[n]
            diff = theta[w_idx] - theta[l_idx] + h * home_flags
            sig = 1.0 / (1.0 + np.exp(diff))  # 1 - sigmoid(diff)

            g = np.zeros(n + 1)
            # gradient wrt theta[i]: sum over games where i is winner of -sig, plus games where i is loser of +sig
            np.add.at(g[:n], w_idx, -sig)
            np.add.at(g[:n], l_idx, sig)
            g[:n] += 2 * l2_reg * theta
            g[n] = np.sum(-sig * home_flags)
            return g

        res = minimize(neg_log_lik, theta0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10})

        theta_opt = res.x[:n]
        theta_opt -= theta_opt.mean()  # Center strengths

        for i, t in enumerate(teams):
            strengths[(season, t)] = theta_opt[i]

        if season % 5 == 0 or season >= 2020:
            print(f"  Season {season}: {len(teams)} teams, {len(w_idx)} games, converged={res.success}")

    return strengths


def compute_elo(reg_df, conf_df, k=40, home_adv=64, carryover=0.65):
    """Simplified Elo (same as run_experiment.py)."""
    elo = {}
    conf_map = {}
    for _, r in conf_df.iterrows():
        conf_map[(int(r.Season), int(r.TeamID))] = r.ConfAbbrev

    seasons = sorted(reg_df["Season"].unique())
    for season in seasons:
        if season > seasons[0]:
            conf_elos = {}
            for (s, t), e in elo.items():
                if s == season - 1:
                    c = conf_map.get((season - 1, t), "other")
                    conf_elos.setdefault(c, []).append(e)
            conf_means = {c: np.mean(v) for c, v in conf_elos.items()}
            for (s, t), e in list(elo.items()):
                if s == season - 1:
                    c = conf_map.get((season - 1, t), "other")
                    cm = conf_means.get(c, 1500)
                    elo[(season, t)] = carryover * e + (1 - carryover) * cm

        sg = reg_df[reg_df.Season == season].sort_values("DayNum")
        for _, g in sg.iterrows():
            wt, lt = int(g.WTeamID), int(g.LTeamID)
            ew = elo.get((season, wt), 1500)
            el_ = elo.get((season, lt), 1500)
            ha = home_adv if g.WLoc == "H" else (-home_adv if g.WLoc == "A" else 0)
            exp_w = 1 / (1 + 10 ** (-(ew - el_ + ha) / 400))
            mov = abs(int(g.WScore) - int(g.LScore))
            mov_mult = ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(ew - el_))
            shift = k * mov_mult * (1 - exp_w)
            elo[(season, wt)] = ew + shift
            elo[(season, lt)] = el_ - shift

    return elo


def evaluate_strength_metric(strengths, tourney_df, seed_df, label="BT"):
    """Evaluate a strength metric (BT or Elo) on tournament prediction via Brier score."""
    # Build seed map
    seed_map = {}
    for _, r in seed_df.iterrows():
        seed_num = int(r.Seed[1:3])
        seed_map[(int(r.Season), int(r.TeamID))] = seed_num

    eval_seasons = sorted(tourney_df["Season"].unique())[-10:]
    scores = []

    for vs in eval_seasons:
        # Get tournament games for this season
        tg = tourney_df[tourney_df.Season == vs]
        if len(tg) == 0:
            continue

        X_list = []
        y_list = []
        for _, g in tg.iterrows():
            wt, lt = int(g.WTeamID), int(g.LTeamID)
            t1, t2 = min(wt, lt), max(wt, lt)
            s1 = strengths.get((vs, t1), 0 if "BT" in label else 1500)
            s2 = strengths.get((vs, t2), 0 if "BT" in label else 1500)
            sd1 = seed_map.get((vs, t1), 8)
            sd2 = seed_map.get((vs, t2), 8)
            X_list.append([s1 - s2, sd1 - sd2])
            y_list.append(1 if t1 == wt else 0)

        X = np.array(X_list)
        y = np.array(y_list)

        # Train on prior seasons, predict on this season
        X_train_list, y_train_list = [], []
        train_seasons = [s for s in eval_seasons if s < vs]
        if len(train_seasons) == 0:
            continue

        for ts in train_seasons:
            tgt = tourney_df[tourney_df.Season == ts]
            for _, g in tgt.iterrows():
                wt, lt = int(g.WTeamID), int(g.LTeamID)
                t1, t2 = min(wt, lt), max(wt, lt)
                s1 = strengths.get((ts, t1), 0 if "BT" in label else 1500)
                s2 = strengths.get((ts, t2), 0 if "BT" in label else 1500)
                sd1 = seed_map.get((ts, t1), 8)
                sd2 = seed_map.get((ts, t2), 8)
                X_train_list.append([s1 - s2, sd1 - sd2])
                y_train_list.append(1 if t1 == wt else 0)

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_train, y_train)
        preds = lr.predict_proba(X)[:, 1]
        brier = brier_score_loss(y, preds)
        scores.append(brier)
        print(f"  {vs}: n={len(y)}, {label} Brier={brier:.5f}")

    mean_brier = np.mean(scores)
    print(f"  Mean {label} Brier (last 10 seasons): {mean_brier:.5f}")
    return mean_brier


def main():
    print("=" * 60)
    print("Bradley-Terry vs Elo Standalone Evaluation")
    print("=" * 60)

    # Load data
    m_reg = pd.read_csv(DATA_DIR / "MRegularSeasonCompactResults.csv")
    w_reg = pd.read_csv(DATA_DIR / "WRegularSeasonCompactResults.csv")
    m_conf = pd.read_csv(DATA_DIR / "MTeamConferences.csv")
    w_conf = pd.read_csv(DATA_DIR / "WTeamConferences.csv")
    m_tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    w_tourney = pd.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")
    m_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    w_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")

    reg = pd.concat([m_reg, w_reg], ignore_index=True)
    conf = pd.concat([m_conf, w_conf], ignore_index=True)
    tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)
    seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

    print(f"\nData: {len(reg)} regular season games, {len(tourney)} tournament games")

    # Compute BT strengths
    print("\n--- Computing Bradley-Terry strengths ---")
    bt_strengths = compute_bt_strengths(reg, conf, carryover=0.65, l2_reg=0.01)
    print(f"Total BT ratings: {len(bt_strengths)}")

    # Compute Elo
    print("\n--- Computing Elo ratings ---")
    elo_ratings = compute_elo(reg, conf, k=40, home_adv=64, carryover=0.65)
    print(f"Total Elo ratings: {len(elo_ratings)}")

    # Check correlation
    common_keys = set(bt_strengths.keys()) & set(elo_ratings.keys())
    recent_keys = [(s, t) for s, t in common_keys if s >= 2015]
    bt_vals = [bt_strengths[k] for k in recent_keys]
    elo_vals = [elo_ratings[k] for k in recent_keys]
    corr = np.corrcoef(bt_vals, elo_vals)[0, 1]
    print(f"\nBT-Elo correlation (2015+): {corr:.5f}")

    # Evaluate both
    print("\n--- Evaluating BT (last 10 tournament seasons) ---")
    bt_brier = evaluate_strength_metric(bt_strengths, tourney, seeds, "BT")

    print("\n--- Evaluating Elo (last 10 tournament seasons) ---")
    elo_brier = evaluate_strength_metric(elo_ratings, tourney, seeds, "Elo")

    print("\n" + "=" * 60)
    print(f"RESULT: BT Brier = {bt_brier:.5f}, Elo Brier = {elo_brier:.5f}")
    print(f"Delta: {bt_brier - elo_brier:+.5f} ({'BT worse' if bt_brier > elo_brier else 'BT better'})")
    if bt_brier < 0.18:
        print("VERDICT: BT passes threshold (< 0.18). Proceed with integration.")
    else:
        print("VERDICT: BT too weak. Abandon experiment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
