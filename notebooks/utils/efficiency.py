"""KenPom-style efficiency metrics for NCAA basketball.

Calculates offensive/defensive efficiency, the Four Factors, and other
advanced stats from detailed box score data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_possessions(
    fga: pd.Series,
    orb: pd.Series,
    to: pd.Series,
    fta: pd.Series,
) -> pd.Series:
    """Estimate possessions from box score stats.

    Possessions = FGA - ORB + TO + 0.475 * FTA
    """
    return fga - orb + to + 0.475 * fta


def calculate_game_stats(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-game advanced stats from detailed results.

    Takes the wide format (W/L prefixed columns) and creates two rows per game
    (one per team) with team-centric stats.

    Returns:
        DataFrame with columns: Season, DayNum, TeamID, OpponentID,
        Score, OppScore, Won, Loc, Poss, OppPoss,
        OffEff, DefEff, NetEff, eFG, OppeFG, TORate, OppTORate,
        ORBRate, OppORBRate, FTRate, OppFTRate, FG3Rate, OppFG3Rate
    """
    rows = []

    for _, g in detailed_df.iterrows():
        season = g["Season"]
        day = g["DayNum"]

        # Winner stats
        w_poss = estimate_possessions(g["WFGA"], g["WOR"], g["WTO"], g["WFTA"])
        l_poss = estimate_possessions(g["LFGA"], g["LOR"], g["LTO"], g["LFTA"])

        # Avoid division by zero
        w_poss = max(w_poss, 1)
        l_poss = max(l_poss, 1)

        w_efg = (g["WFGM"] + 0.5 * g["WFGM3"]) / max(g["WFGA"], 1)
        l_efg = (g["LFGM"] + 0.5 * g["LFGM3"]) / max(g["LFGA"], 1)
        w_to_rate = g["WTO"] / w_poss
        l_to_rate = g["LTO"] / l_poss
        w_orb_rate = g["WOR"] / max(g["WOR"] + g["LDR"], 1)
        l_orb_rate = g["LOR"] / max(g["LOR"] + g["WDR"], 1)
        w_ft_rate = g["WFTA"] / max(g["WFGA"], 1)
        l_ft_rate = g["LFTA"] / max(g["LFGA"], 1)
        w_fg3_rate = g["WFGM3"] / max(g["WFGA"], 1)
        l_fg3_rate = g["LFGM3"] / max(g["LFGA"], 1)

        w_off_eff = g["WScore"] / w_poss * 100
        w_def_eff = g["LScore"] / l_poss * 100
        l_off_eff = g["LScore"] / l_poss * 100
        l_def_eff = g["WScore"] / w_poss * 100

        # Determine location for each team
        w_loc = g["WLoc"]
        if w_loc == "H":
            l_loc = "A"
        elif w_loc == "A":
            l_loc = "H"
        else:
            l_loc = "N"

        # Winner row
        rows.append({
            "Season": season, "DayNum": day,
            "TeamID": g["WTeamID"], "OpponentID": g["LTeamID"],
            "Score": g["WScore"], "OppScore": g["LScore"],
            "Won": 1, "Loc": w_loc,
            "Poss": w_poss, "OppPoss": l_poss,
            "OffEff": w_off_eff, "DefEff": w_def_eff,
            "NetEff": w_off_eff - w_def_eff,
            "eFG": w_efg, "OppeFG": l_efg,
            "TORate": w_to_rate, "OppTORate": l_to_rate,
            "ORBRate": w_orb_rate, "OppORBRate": l_orb_rate,
            "FTRate": w_ft_rate, "OppFTRate": l_ft_rate,
            "FG3Rate": w_fg3_rate, "OppFG3Rate": l_fg3_rate,
            "FGM": g["WFGM"], "FGA": g["WFGA"],
            "FGM3": g["WFGM3"], "FGA3": g["WFGA3"],
            "FTM": g["WFTM"], "FTA": g["WFTA"],
            "ORB": g["WOR"], "DRB": g["WDR"],
            "Ast": g["WAst"], "TO": g["WTO"],
            "Stl": g["WStl"], "Blk": g["WBlk"],
            "PF": g["WPF"],
        })

        # Loser row
        rows.append({
            "Season": season, "DayNum": day,
            "TeamID": g["LTeamID"], "OpponentID": g["WTeamID"],
            "Score": g["LScore"], "OppScore": g["WScore"],
            "Won": 0, "Loc": l_loc,
            "Poss": l_poss, "OppPoss": w_poss,
            "OffEff": l_off_eff, "DefEff": l_def_eff,
            "NetEff": l_off_eff - l_def_eff,
            "eFG": l_efg, "OppeFG": w_efg,
            "TORate": l_to_rate, "OppTORate": w_to_rate,
            "ORBRate": l_orb_rate, "OppORBRate": w_orb_rate,
            "FTRate": l_ft_rate, "OppFTRate": w_ft_rate,
            "FG3Rate": l_fg3_rate, "OppFG3Rate": w_fg3_rate,
            "FGM": g["LFGM"], "FGA": g["LFGA"],
            "FGM3": g["LFGM3"], "FGA3": g["LFGA3"],
            "FTM": g["LFTM"], "FTA": g["LFTA"],
            "ORB": g["LOR"], "DRB": g["LDR"],
            "Ast": g["LAst"], "TO": g["LTO"],
            "Stl": g["LStl"], "Blk": g["LBlk"],
            "PF": g["LPF"],
        })

    return pd.DataFrame(rows)


def aggregate_season_stats(
    game_stats: pd.DataFrame,
    max_day: int = 132,
) -> pd.DataFrame:
    """Aggregate game-level stats into season-level team features.

    Only uses regular season games (DayNum <= max_day) to avoid
    tournament data leakage.

    Returns:
        DataFrame with one row per (Season, TeamID) containing aggregated stats.
    """
    # Filter to regular season only
    reg = game_stats[game_stats.DayNum <= max_day].copy()

    agg_funcs = {
        "Won": ["sum", "count"],
        "Score": "mean",
        "OppScore": "mean",
        "Poss": "mean",
        "OffEff": ["mean", "std"],
        "DefEff": ["mean", "std"],
        "NetEff": "mean",
        "eFG": "mean",
        "OppeFG": "mean",
        "TORate": "mean",
        "OppTORate": "mean",
        "ORBRate": "mean",
        "OppORBRate": "mean",
        "FTRate": "mean",
        "OppFTRate": "mean",
        "FG3Rate": "mean",
        "OppFG3Rate": "mean",
    }

    grouped = reg.groupby(["Season", "TeamID"]).agg(agg_funcs)

    # Flatten multi-level columns
    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
        for col in grouped.columns
    ]
    grouped = grouped.reset_index()

    # Rename for clarity
    grouped = grouped.rename(columns={
        "Won_sum": "Wins",
        "Won_count": "Games",
        "Score_mean": "AvgScore",
        "OppScore_mean": "AvgOppScore",
        "Poss_mean": "AvgPoss",
        "OffEff_mean": "OffEff",
        "OffEff_std": "OffEff_std",
        "DefEff_mean": "DefEff",
        "DefEff_std": "DefEff_std",
        "NetEff_mean": "NetEff",
        "eFG_mean": "eFG",
        "OppeFG_mean": "OppeFG",
        "TORate_mean": "TORate",
        "OppTORate_mean": "OppTORate",
        "ORBRate_mean": "ORBRate",
        "OppORBRate_mean": "OppORBRate",
        "FTRate_mean": "FTRate",
        "OppFTRate_mean": "OppFTRate",
        "FG3Rate_mean": "FG3Rate",
        "OppFG3Rate_mean": "OppFG3Rate",
    })

    # Derived features
    grouped["WinPct"] = grouped["Wins"] / grouped["Games"]
    grouped["AvgMargin"] = grouped["AvgScore"] - grouped["AvgOppScore"]

    # Four Factors differentials (offense vs opponent offense)
    grouped["eFG_diff"] = grouped["eFG"] - grouped["OppeFG"]
    grouped["TORate_diff"] = grouped["OppTORate"] - grouped["TORate"]  # Positive = good (opponent has more TOs)
    grouped["ORBRate_diff"] = grouped["ORBRate"] - grouped["OppORBRate"]
    grouped["FTRate_diff"] = grouped["FTRate"] - grouped["OppFTRate"]

    return grouped


def compute_last_n_games(
    game_stats: pd.DataFrame,
    n: int = 10,
    max_day: int = 132,
) -> pd.DataFrame:
    """Compute stats from last N regular season games (recent form).

    Returns:
        DataFrame with Season, TeamID, and last-N-game stats.
    """
    reg = game_stats[game_stats.DayNum <= max_day].copy()
    reg = reg.sort_values(["Season", "TeamID", "DayNum"])

    # Take last N games per team per season
    last_n = reg.groupby(["Season", "TeamID"]).tail(n)

    agg = last_n.groupby(["Season", "TeamID"]).agg({
        "Won": "mean",
        "OffEff": "mean",
        "DefEff": "mean",
        "NetEff": "mean",
        "eFG": "mean",
        "Score": "mean",
        "OppScore": "mean",
    }).reset_index()

    agg = agg.rename(columns={
        "Won": f"WinPct_L{n}",
        "OffEff": f"OffEff_L{n}",
        "DefEff": f"DefEff_L{n}",
        "NetEff": f"NetEff_L{n}",
        "eFG": f"eFG_L{n}",
        "Score": f"AvgScore_L{n}",
        "OppScore": f"AvgOppScore_L{n}",
    })

    return agg


def build_efficiency_features(
    detailed_df: pd.DataFrame,
    max_day: int = 132,
    last_n_games: int = 10,
) -> pd.DataFrame:
    """End-to-end pipeline: detailed box scores -> team season features.

    Args:
        detailed_df: MRegularSeasonDetailedResults or WRegularSeasonDetailedResults
        max_day: DayNum cutoff for regular season
        last_n_games: Number of recent games for form features

    Returns:
        DataFrame with Season, TeamID, and all efficiency features.
    """
    game_stats = calculate_game_stats(detailed_df)
    season_stats = aggregate_season_stats(game_stats, max_day=max_day)
    form_stats = compute_last_n_games(game_stats, n=last_n_games, max_day=max_day)

    # Merge season-level and form features
    result = season_stats.merge(form_stats, on=["Season", "TeamID"], how="left")

    return result
