"""Custom Elo rating system for NCAA basketball.

Designed for March Machine Learning Mania competition. Processes regular season
and tournament games chronologically, producing team-level Elo ratings.

Key design decisions based on FiveThirtyEight / competition winner approaches:
- K-factor: ~32 (tunable)
- Home court advantage: ~100 Elo points (~3.5 point spread)
- Margin of victory multiplier to weight blowouts vs close games
- Season reset: regress toward conference mean between seasons
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EloSystem:
    """NCAA Basketball Elo Rating System."""

    def __init__(
        self,
        k: float = 32,
        home_advantage: float = 100,
        season_carryover: float = 0.65,
        mean_rating: float = 1500,
        mov_exponent: float = 0.8,
        mov_denom_base: float = 7.5,
        mov_denom_elo_coeff: float = 0.006,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.season_carryover = season_carryover
        self.mean_rating = mean_rating
        self.mov_exponent = mov_exponent
        self.mov_denom_base = mov_denom_base
        self.mov_denom_elo_coeff = mov_denom_elo_coeff

        self.ratings: dict[tuple[int, int], float] = {}  # (season, team_id) -> elo
        self.history: list[dict] = []

    def expected_win_prob(self, elo_a: float, elo_b: float) -> float:
        """Win probability for team A given Elo ratings."""
        return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))

    def _mov_multiplier(self, mov: int, elo_diff: float) -> float:
        """Margin of victory multiplier. Dampens blowout impact."""
        return ((abs(mov) + 3.0) ** self.mov_exponent) / (
            self.mov_denom_base + self.mov_denom_elo_coeff * abs(elo_diff)
        )

    def _get_rating(self, season: int, team_id: int) -> float:
        """Get team's current Elo (default: mean_rating for new teams)."""
        return self.ratings.get((season, team_id), self.mean_rating)

    def update(
        self,
        season: int,
        team_w: int,
        team_l: int,
        score_w: int,
        score_l: int,
        location: str,
        day_num: int = 0,
    ) -> tuple[float, float]:
        """Update ratings after a game. Returns (new_elo_w, new_elo_l)."""
        elo_w = self._get_rating(season, team_w)
        elo_l = self._get_rating(season, team_l)

        # Adjust for home court
        if location == "H":  # Winner was home
            elo_w_adj = elo_w + self.home_advantage
            elo_l_adj = elo_l
        elif location == "A":  # Winner was away
            elo_w_adj = elo_w
            elo_l_adj = elo_l + self.home_advantage
        else:  # Neutral
            elo_w_adj = elo_w
            elo_l_adj = elo_l

        # Expected win probability for winner
        exp_w = self.expected_win_prob(elo_w_adj, elo_l_adj)

        # Margin of victory
        mov = score_w - score_l
        mov_mult = self._mov_multiplier(mov, elo_w - elo_l)

        # Update ratings
        shift = self.k * mov_mult * (1.0 - exp_w)
        new_elo_w = elo_w + shift
        new_elo_l = elo_l - shift

        self.ratings[(season, team_w)] = new_elo_w
        self.ratings[(season, team_l)] = new_elo_l

        self.history.append({
            "season": season,
            "day_num": day_num,
            "team_w": team_w,
            "team_l": team_l,
            "elo_w_before": elo_w,
            "elo_l_before": elo_l,
            "elo_w_after": new_elo_w,
            "elo_l_after": new_elo_l,
            "expected_w": exp_w,
            "mov": mov,
            "shift": shift,
        })

        return new_elo_w, new_elo_l

    def reset_season(
        self,
        prev_season: int,
        new_season: int,
        team_conferences: dict[int, str],
    ) -> None:
        """Between-season reset: regress toward conference mean."""
        # Compute conference averages from previous season
        conf_ratings: dict[str, list[float]] = {}
        for team_id, conf in team_conferences.items():
            elo = self._get_rating(prev_season, team_id)
            conf_ratings.setdefault(conf, []).append(elo)

        conf_means = {c: np.mean(rs) for c, rs in conf_ratings.items()}

        # Reset each team's rating
        for team_id, conf in team_conferences.items():
            prev_elo = self._get_rating(prev_season, team_id)
            conf_mean = conf_means.get(conf, self.mean_rating)
            new_elo = (
                self.season_carryover * prev_elo
                + (1 - self.season_carryover) * conf_mean
            )
            self.ratings[(new_season, team_id)] = new_elo

    def process_all_games(
        self,
        regular_season_df: pd.DataFrame,
        conferences_df: pd.DataFrame,
        tourney_df: pd.DataFrame | None = None,
        max_regular_day: int = 132,
    ) -> pd.DataFrame:
        """Process all games chronologically across all seasons.

        Args:
            regular_season_df: DataFrame with Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc
            conferences_df: DataFrame with Season, TeamID, ConfAbbrev
            tourney_df: Optional tournament results (processed separately for training)
            max_regular_day: DayNum cutoff for regular season (before tournament)

        Returns:
            DataFrame of team Elo ratings at end of regular season per season.
        """
        # Combine and sort all games
        all_games = regular_season_df.copy()
        if tourney_df is not None:
            all_games = pd.concat([all_games, tourney_df], ignore_index=True)

        all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)
        seasons = sorted(all_games["Season"].unique())

        # Track end-of-regular-season ratings
        eor_ratings = []

        for season in seasons:
            # Season reset (except first season)
            if season > seasons[0]:
                conf_map = dict(
                    zip(
                        conferences_df[conferences_df.Season == season]["TeamID"],
                        conferences_df[conferences_df.Season == season]["ConfAbbrev"],
                    )
                )
                if conf_map:
                    self.reset_season(season - 1, season, conf_map)

            # Process games in chronological order
            season_games = all_games[all_games.Season == season]

            for _, game in season_games.iterrows():
                self.update(
                    season=season,
                    team_w=int(game.WTeamID),
                    team_l=int(game.LTeamID),
                    score_w=int(game.WScore),
                    score_l=int(game.LScore),
                    location=str(game.WLoc),
                    day_num=int(game.DayNum),
                )

            # Capture end-of-regular-season ratings
            reg_games = season_games[season_games.DayNum <= max_regular_day]
            if len(reg_games) > 0:
                # Get all teams that played this season
                teams = set(reg_games["WTeamID"].unique()) | set(
                    reg_games["LTeamID"].unique()
                )
                for team_id in teams:
                    eor_ratings.append({
                        "Season": season,
                        "TeamID": int(team_id),
                        "Elo": self._get_rating(season, int(team_id)),
                    })

        return pd.DataFrame(eor_ratings)

    def get_matchup_prob(
        self, season: int, team_a: int, team_b: int
    ) -> float:
        """Get predicted win probability for team_a vs team_b (neutral court)."""
        elo_a = self._get_rating(season, team_a)
        elo_b = self._get_rating(season, team_b)
        return self.expected_win_prob(elo_a, elo_b)


def build_elo_features(
    m_reg_compact: pd.DataFrame,
    m_team_conf: pd.DataFrame,
    w_reg_compact: pd.DataFrame,
    w_team_conf: pd.DataFrame,
    m_tourney: pd.DataFrame | None = None,
    w_tourney: pd.DataFrame | None = None,
    **elo_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, EloSystem, EloSystem]:
    """Build Elo features for both men's and women's teams.

    Returns:
        (m_elo_df, w_elo_df, m_elo_system, w_elo_system)
        Each elo_df has columns: Season, TeamID, Elo
    """
    m_elo = EloSystem(**elo_kwargs)
    w_elo = EloSystem(**elo_kwargs)

    m_elo_df = m_elo.process_all_games(m_reg_compact, m_team_conf, m_tourney)
    w_elo_df = w_elo.process_all_games(w_reg_compact, w_team_conf, w_tourney)

    return m_elo_df, w_elo_df, m_elo, w_elo
