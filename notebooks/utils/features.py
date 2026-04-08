"""Feature engineering pipeline for March Machine Learning Mania.

Combines Elo, efficiency, seed, Massey ordinals, and conference features
into a unified matchup-level training dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Seed Features ─────────────────────────────────────────────────


def parse_seed(seed_str: str) -> int:
    """Parse seed string 'W01' -> 1, 'X16a' -> 16."""
    return int(seed_str[1:3])


def build_seed_features(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric seed column to seeds DataFrame.

    Returns DataFrame with: Season, TeamID, Seed (original string), SeedNum.
    """
    df = seeds_df.copy()
    df["SeedNum"] = df["Seed"].apply(parse_seed)
    return df


def compute_historical_seed_win_rates(
    tourney_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute historical win rates for each seed-vs-seed matchup.

    Returns DataFrame with columns: StrongSeed, WeakSeed, HistWinRate, HistGames.
    """
    seeds = build_seed_features(seeds_df)

    games = tourney_df.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "WTeamID", "SeedNum": "WSeed"}
        ),
        on=["Season", "WTeamID"],
    ).merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "LTeamID", "SeedNum": "LSeed"}
        ),
        on=["Season", "LTeamID"],
    )

    games["StrongSeed"] = games[["WSeed", "LSeed"]].min(axis=1)
    games["WeakSeed"] = games[["WSeed", "LSeed"]].max(axis=1)
    games["StrongWon"] = (games["WSeed"] == games["StrongSeed"]).astype(int)

    stats = (
        games.groupby(["StrongSeed", "WeakSeed"])
        .agg(HistGames=("StrongWon", "count"), StrongWins=("StrongWon", "sum"))
        .reset_index()
    )
    stats["HistWinRate"] = stats["StrongWins"] / stats["HistGames"]

    return stats[["StrongSeed", "WeakSeed", "HistWinRate", "HistGames"]]


def get_seed_matchup_win_rate(
    seed_a: int,
    seed_b: int,
    hist_rates: pd.DataFrame,
    default: float = 0.5,
) -> float:
    """Look up historical win rate for seed_a vs seed_b.

    Returns P(seed_a wins) based on historical data.
    """
    strong = min(seed_a, seed_b)
    weak = max(seed_a, seed_b)

    if strong == weak:
        return 0.5

    row = hist_rates[(hist_rates.StrongSeed == strong) & (hist_rates.WeakSeed == weak)]
    if len(row) == 0:
        return default

    rate = row["HistWinRate"].values[0]
    # Return rate from perspective of seed_a
    if seed_a == strong:
        return rate
    else:
        return 1.0 - rate


# ── Massey Ordinals ───────────────────────────────────────────────


def aggregate_massey_ordinals(
    massey_df: pd.DataFrame,
    max_day: int = 133,
    top_systems: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate Massey ordinal rankings for each team.

    Uses the latest pre-tournament ranking day for each system.

    Args:
        massey_df: MMasseyOrdinals DataFrame.
        max_day: Maximum DayNum (before tournament starts).
        top_systems: If provided, only use these ranking systems.

    Returns:
        DataFrame with: Season, TeamID, MasseyMean, MasseyMedian,
        MasseyMin, MasseyMax, MasseyStd, MasseyCount.
    """
    df = massey_df[massey_df.RankingDayNum <= max_day].copy()

    if top_systems:
        df = df[df.SystemName.isin(top_systems)]

    # Get latest ranking day per system per season per team
    latest = df.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].max()
    latest = latest.reset_index()
    df = df.merge(latest, on=["Season", "SystemName", "TeamID", "RankingDayNum"])

    # Aggregate across systems
    agg = (
        df.groupby(["Season", "TeamID"])["OrdinalRank"]
        .agg(["mean", "median", "min", "max", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "Season", "TeamID",
        "MasseyMean", "MasseyMedian", "MasseyMin", "MasseyMax",
        "MasseyStd", "MasseyCount",
    ]
    agg["MasseyStd"] = agg["MasseyStd"].fillna(0)

    return agg


# ── Conference Features ──────────────────────────────────────────


def build_conference_features(
    team_conf_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build conference-level features from Elo ratings.

    Returns DataFrame with: Season, TeamID, ConfAbbrev,
    ConfEloMean, ConfEloStd, ConfRank, IsPowerConf.
    """
    POWER_CONFS = {"acc", "big_twelve", "big_ten", "sec", "big_east", "pac_twelve", "aac"}

    # Merge conference info with Elo
    df = team_conf_df.merge(elo_df, on=["Season", "TeamID"], how="left")

    # Conference-level stats
    conf_stats = (
        df.groupby(["Season", "ConfAbbrev"])["Elo"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    conf_stats.columns = ["Season", "ConfAbbrev", "ConfEloMean", "ConfEloStd", "ConfSize"]
    conf_stats["ConfEloStd"] = conf_stats["ConfEloStd"].fillna(0)

    # Rank conferences by mean Elo per season
    conf_stats["ConfRank"] = conf_stats.groupby("Season")["ConfEloMean"].rank(
        ascending=False
    )

    # Merge back to team level
    result = team_conf_df.merge(conf_stats, on=["Season", "ConfAbbrev"], how="left")
    result["IsPowerConf"] = result["ConfAbbrev"].isin(POWER_CONFS).astype(int)

    return result


# ── Win/Loss Record Features ─────────────────────────────────────


def build_record_features(
    compact_df: pd.DataFrame,
    max_day: int = 132,
) -> pd.DataFrame:
    """Build win/loss record features from compact results.

    Returns DataFrame with: Season, TeamID, Wins, Losses, WinPct,
    AvgMargin, HomeWins, AwayWins, NeutralWins.
    """
    reg = compact_df[compact_df.DayNum <= max_day].copy()
    reg["Margin"] = reg["WScore"] - reg["LScore"]

    # Winner records
    w_stats = (
        reg.groupby(["Season", "WTeamID"])
        .agg(
            Wins=("Margin", "count"),
            TotalMargin=("Margin", "sum"),
            HomeWins=("WLoc", lambda x: (x == "H").sum()),
            AwayWins=("WLoc", lambda x: (x == "A").sum()),
            NeutralWins=("WLoc", lambda x: (x == "N").sum()),
        )
        .reset_index()
        .rename(columns={"WTeamID": "TeamID"})
    )

    # Loser records
    l_stats = (
        reg.groupby(["Season", "LTeamID"])
        .agg(Losses=("Margin", "count"), TotalMarginLost=("Margin", "sum"))
        .reset_index()
        .rename(columns={"LTeamID": "TeamID"})
    )

    # Merge
    records = w_stats.merge(l_stats, on=["Season", "TeamID"], how="outer").fillna(0)
    records["Games"] = records["Wins"] + records["Losses"]
    records["WinPct_Record"] = records["Wins"] / records["Games"].clip(lower=1)
    records["AvgMargin_Record"] = (
        records["TotalMargin"] - records["TotalMarginLost"]
    ) / records["Games"].clip(lower=1)

    return records[
        ["Season", "TeamID", "Wins", "Losses", "Games", "WinPct_Record",
         "AvgMargin_Record", "HomeWins", "AwayWins", "NeutralWins"]
    ]


# ── Matchup Feature Builder ──────────────────────────────────────


def build_team_features(
    elo_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    massey_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    record_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all team-level features into a single DataFrame.

    Returns DataFrame indexed by (Season, TeamID) with all features.
    """
    # Start with Elo
    result = elo_df[["Season", "TeamID", "Elo"]].copy()

    # Merge efficiency
    if efficiency_df is not None and len(efficiency_df) > 0:
        result = result.merge(efficiency_df, on=["Season", "TeamID"], how="left")

    # Merge seeds
    seeds = build_seed_features(seeds_df)
    result = result.merge(
        seeds[["Season", "TeamID", "SeedNum"]], on=["Season", "TeamID"], how="left"
    )

    # Merge Massey
    if massey_df is not None and len(massey_df) > 0:
        result = result.merge(massey_df, on=["Season", "TeamID"], how="left")

    # Merge conference
    if conf_df is not None and len(conf_df) > 0:
        conf_cols = [c for c in conf_df.columns if c not in ["ConfAbbrev"]]
        result = result.merge(conf_df[conf_cols], on=["Season", "TeamID"], how="left")

    # Merge records
    if record_df is not None and len(record_df) > 0:
        result = result.merge(record_df, on=["Season", "TeamID"], how="left")

    return result


def build_matchup_features(
    team_features: pd.DataFrame,
    team1_id: int,
    team2_id: int,
    season: int,
    feature_cols: list[str] | None = None,
) -> dict:
    """Create matchup-level features for a single team pair.

    Computes differences and ratios of all numeric features.
    Convention: TeamID1 < TeamID2 (to match submission format).

    Returns dict of matchup features.
    """
    t1 = team_features[
        (team_features.Season == season) & (team_features.TeamID == team1_id)
    ]
    t2 = team_features[
        (team_features.Season == season) & (team_features.TeamID == team2_id)
    ]

    if len(t1) == 0 or len(t2) == 0:
        return {}

    t1 = t1.iloc[0]
    t2 = t2.iloc[0]

    if feature_cols is None:
        feature_cols = [
            c for c in team_features.columns
            if c not in ["Season", "TeamID"] and team_features[c].dtype in ["float64", "int64", "float32", "int32"]
        ]

    matchup = {"Season": season, "TeamID1": team1_id, "TeamID2": team2_id}

    for col in feature_cols:
        v1 = t1.get(col, np.nan)
        v2 = t2.get(col, np.nan)
        if pd.notna(v1) and pd.notna(v2):
            matchup[f"{col}_diff"] = v1 - v2
        else:
            matchup[f"{col}_diff"] = np.nan

    return matchup


def build_training_data(
    tourney_df: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build matchup-level training data from historical tournament games.

    For each game, orients so TeamID1 < TeamID2 to match submission format.
    Target: 1 if TeamID1 won, 0 otherwise.

    Returns DataFrame with matchup features and target column.
    """
    rows = []

    for _, game in tourney_df.iterrows():
        season = int(game.Season)
        team1 = int(min(game.WTeamID, game.LTeamID))
        team2 = int(max(game.WTeamID, game.LTeamID))
        target = 1 if int(game.WTeamID) == team1 else 0

        matchup = build_matchup_features(
            team_features, team1, team2, season, feature_cols
        )
        if matchup:
            matchup["target"] = target
            rows.append(matchup)

    return pd.DataFrame(rows)


def build_submission_predictions(
    sample_sub: pd.DataFrame,
    team_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build feature matrix for all submission matchups.

    Returns DataFrame with same matchup features (no target).
    """
    rows = []

    for _, row in sample_sub.iterrows():
        parts = row["ID"].split("_")
        season = int(parts[0])
        team1 = int(parts[1])
        team2 = int(parts[2])

        matchup = build_matchup_features(
            team_features, team1, team2, season, feature_cols
        )
        if matchup:
            matchup["ID"] = row["ID"]
            rows.append(matchup)

    return pd.DataFrame(rows)
