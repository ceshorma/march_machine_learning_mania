"""
Build TravelDistance and RestDays features for tournament prediction.
- RestDays: days since team's last game before tournament starts
- TravelDistance: distance (km) from team's home city to tournament venue

Outputs: data/team_travel_rest.csv with columns:
  Season, TeamID, RestDays, TravelDist_km
"""
import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def geocode_cities(cities_df):
    """Geocode cities using Nominatim (free, no API key needed). Returns dict CityID -> (lat, lon)."""
    import urllib.request
    import urllib.parse
    import json
    import time

    coords = {}
    failed = []
    cache_path = DATA_DIR / "city_coords_cache.json"

    # Load cache if exists
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached coordinates")
    else:
        cache = {}

    for _, row in cities_df.iterrows():
        city_id = str(row["CityID"])
        city = row["City"]
        state = row["State"]
        cache_key = f"{city},{state}"

        if cache_key in cache:
            coords[int(city_id)] = tuple(cache[cache_key])
            continue

        # Query Nominatim
        query = urllib.parse.urlencode({
            "q": f"{city}, {state}, USA",
            "format": "json",
            "limit": 1
        })
        url = f"https://nominatim.openstreetmap.org/search?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "MarchMadnessML/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                coords[int(city_id)] = (lat, lon)
                cache[cache_key] = [lat, lon]
            else:
                failed.append((city_id, city, state))
        except Exception as e:
            failed.append((city_id, city, state))

        time.sleep(1.1)  # Nominatim rate limit: 1 req/sec

    # Save cache
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    if failed:
        print(f"  Failed to geocode {len(failed)} cities: {failed[:10]}...")

    return coords


def find_home_cities(game_cities_df, reg_df, city_coords):
    """Determine each team's home city based on where they play home games."""
    # Merge game cities with results to know which team was home
    merged = game_cities_df[game_cities_df.CRType == "Regular"].merge(
        reg_df[["Season", "DayNum", "WTeamID", "LTeamID", "WLoc"]],
        on=["Season", "DayNum", "WTeamID", "LTeamID"],
        how="inner"
    )

    team_home = {}  # (season, team_id) -> city_id

    # For home games (WLoc=H), winner is at home city
    home_games_w = merged[merged.WLoc == "H"]
    # For away games (WLoc=A), loser is at their home city... no, loser is away.
    # When WLoc=H: winner's home = CityID
    # When WLoc=A: loser's home = CityID (winner traveled there)

    # Count home game occurrences per team per city
    home_counts = {}
    for _, g in home_games_w.iterrows():
        key = (g.Season, int(g.WTeamID))
        city = int(g.CityID)
        home_counts.setdefault(key, {})
        home_counts[key][city] = home_counts[key].get(city, 0) + 1

    # Away team's home: when WLoc=A, WTeamID traveled, LTeamID is at home
    away_as_home = merged[merged.WLoc == "A"]
    for _, g in away_as_home.iterrows():
        key = (g.Season, int(g.LTeamID))
        city = int(g.CityID)
        home_counts.setdefault(key, {})
        home_counts[key][city] = home_counts[key].get(city, 0) + 1

    # Pick most frequent city as home
    for key, counts in home_counts.items():
        best_city = max(counts, key=counts.get)
        if best_city in city_coords:
            team_home[key] = best_city

    return team_home


def compute_rest_days(reg_df, tourney_df):
    """Compute days of rest before first tournament game for each team."""
    rest = {}

    for season in sorted(tourney_df["Season"].unique()):
        # Get all teams in tournament this season
        st = tourney_df[tourney_df.Season == season]
        tourney_teams = set(st["WTeamID"].astype(int)) | set(st["LTeamID"].astype(int))

        # First tournament day
        first_tourney_day = st["DayNum"].min()

        # For each tournament team, find their last regular season game
        sr = reg_df[reg_df.Season == season]
        for tid in tourney_teams:
            team_games = sr[(sr.WTeamID == tid) | (sr.LTeamID == tid)]
            if len(team_games) > 0:
                last_day = team_games["DayNum"].max()
                rest[(season, tid)] = int(first_tourney_day - last_day)
            else:
                rest[(season, tid)] = 7  # default

    return rest


def compute_travel_distances(tourney_df, game_cities_df, team_home, city_coords):
    """Compute travel distance for each team in each tournament season."""
    # Get tournament venue cities
    tourney_cities = game_cities_df[game_cities_df.CRType.str.contains("NCAA", na=False)]

    # For each team-season, compute average distance to tournament venues
    # Since we predict ALL matchups, we use average tournament venue distance as proxy
    travel = {}

    for season in sorted(tourney_df["Season"].unique()):
        st = tourney_df[tourney_df.Season == season]
        tourney_teams = set(st["WTeamID"].astype(int)) | set(st["LTeamID"].astype(int))

        # Get tournament venue cities for this season
        stc = tourney_cities[tourney_cities.Season == season]
        if len(stc) == 0:
            continue

        # Get first-round venue for each team (their first game)
        team_first_venue = {}
        for _, g in st.sort_values("DayNum").iterrows():
            wt, lt = int(g.WTeamID), int(g.LTeamID)
            day = g.DayNum
            # Find matching city
            match = stc[(stc.Season == season) & (stc.DayNum == day) &
                        (stc.WTeamID == g.WTeamID) & (stc.LTeamID == g.LTeamID)]
            if len(match) > 0:
                city_id = int(match.iloc[0].CityID)
                if wt not in team_first_venue:
                    team_first_venue[wt] = city_id
                if lt not in team_first_venue:
                    team_first_venue[lt] = city_id

        # Compute distance from home to first tournament venue
        for tid in tourney_teams:
            home_key = (season, tid)
            if home_key in team_home and tid in team_first_venue:
                home_city = team_home[home_key]
                venue_city = team_first_venue[tid]
                if home_city in city_coords and venue_city in city_coords:
                    h_lat, h_lon = city_coords[home_city]
                    v_lat, v_lon = city_coords[venue_city]
                    dist = haversine_km(h_lat, h_lon, v_lat, v_lon)
                    travel[(season, tid)] = round(dist, 1)

    return travel


def main():
    print("Building Travel Distance and Rest Days features...")

    # Load data
    cities = pd.read_csv(DATA_DIR / "Cities.csv")
    m_gc = pd.read_csv(DATA_DIR / "MGameCities.csv")
    w_gc = pd.read_csv(DATA_DIR / "WGameCities.csv")
    m_reg = pd.read_csv(DATA_DIR / "MRegularSeasonCompactResults.csv")
    w_reg = pd.read_csv(DATA_DIR / "WRegularSeasonCompactResults.csv")
    m_tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    w_tourney = pd.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")

    # Step 1: Geocode cities
    print("\n[1/4] Geocoding cities...")
    city_coords = geocode_cities(cities)
    print(f"  Geocoded {len(city_coords)}/{len(cities)} cities")

    # Step 2: Find home cities
    print("\n[2/4] Finding team home cities...")
    m_home = find_home_cities(m_gc, m_reg, city_coords)
    w_home = find_home_cities(w_gc, w_reg, city_coords)
    team_home = {**m_home, **w_home}
    print(f"  Found home cities for {len(team_home)} team-seasons")

    # Step 3: Compute rest days
    print("\n[3/4] Computing rest days...")
    m_rest = compute_rest_days(m_reg, m_tourney)
    w_rest = compute_rest_days(w_reg, w_tourney)
    rest_days = {**m_rest, **w_rest}
    print(f"  Computed rest days for {len(rest_days)} team-seasons")

    # Rest days stats
    rest_vals = list(rest_days.values())
    print(f"  RestDays: mean={np.mean(rest_vals):.1f}, median={np.median(rest_vals):.0f}, "
          f"min={min(rest_vals)}, max={max(rest_vals)}")

    # Step 4: Compute travel distances
    print("\n[4/4] Computing travel distances...")
    m_travel = compute_travel_distances(m_tourney, m_gc, team_home, city_coords)
    w_travel = compute_travel_distances(w_tourney, w_gc, team_home, city_coords)
    travel = {**m_travel, **w_travel}
    print(f"  Computed travel distance for {len(travel)} team-seasons")

    if travel:
        travel_vals = list(travel.values())
        print(f"  TravelDist: mean={np.mean(travel_vals):.0f}km, median={np.median(travel_vals):.0f}km, "
              f"min={min(travel_vals):.0f}km, max={max(travel_vals):.0f}km")

    # Build output DataFrame
    all_keys = set(rest_days.keys()) | set(travel.keys())
    rows = []
    for (season, tid) in sorted(all_keys):
        rows.append({
            "Season": season,
            "TeamID": tid,
            "RestDays": rest_days.get((season, tid), np.nan),
            "TravelDist_km": travel.get((season, tid), np.nan),
        })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "team_travel_rest.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows)")
    print(f"  Coverage: RestDays={df['RestDays'].notna().sum()}, TravelDist={df['TravelDist_km'].notna().sum()}")


if __name__ == "__main__":
    main()
