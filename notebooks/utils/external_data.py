"""External data loading and team name mapping for March Madness.

Handles data from:
- KenPom (individual efficiency CSVs)
- Barttorvik (individual year CSVs)
- Multisource combined files (KenPom+Barttorvik, EvanMiya, Resumes, 538, Coach)

Key insight: multisource files have `TEAM NO` column = Kaggle `TeamID` directly,
bypassing the lossy team name mapping (~5% unmapped with name-based approach).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Team Name Mapping (for individual kenpom/barttorvik CSVs) ────

EXTERNAL_TO_KAGGLE = {
    "Abilene Christian": "Abilene Chr", "Albany": "SUNY Albany",
    "American": "American Univ", "Appalachian St.": "Appalachian St",
    "Arizona St.": "Arizona St", "Arkansas Little Rock": "Ark Little Rock",
    "Little Rock": "Ark Little Rock", "Arkansas Pine Bluff": "Ark Pine Bluff",
    "Arkansas St.": "Arkansas St", "Alabama St.": "Alabama St",
    "Ball St.": "Ball St", "Boise St.": "Boise St",
    "Boston University": "Boston Univ", "Bowling Green St.": "Bowling Green",
    "Cal St. Bakersfield": "CS Bakersfield", "Cal St. Fullerton": "CS Fullerton",
    "Cal St. Northridge": "CS Northridge", "Cleveland St.": "Cleveland St",
    "Coastal Carolina": "Coastal Car", "College of Charleston": "Col Charleston",
    "Colorado St.": "Colorado St", "Coppin St.": "Coppin St",
    "East Tennessee St.": "ETSU", "Eastern Kentucky": "E Kentucky",
    "Eastern Washington": "E Washington", "Fairleigh Dickinson": "F Dickinson",
    "Florida Atlantic": "FL Atlantic", "Florida Gulf Coast": "FGCU",
    "Florida St.": "Florida St", "Fresno St.": "Fresno St",
    "George Washington": "G Washington", "Georgia St.": "Georgia St",
    "Grambling St.": "Grambling", "Green Bay": "WI Green Bay",
    "Indiana St.": "Indiana St", "Iowa St.": "Iowa St",
    "Jacksonville St.": "Jacksonville St", "Kansas St.": "Kansas St",
    "Kennesaw St.": "Kennesaw", "Kent St.": "Kent",
    "Long Beach St.": "Long Beach St", "Louisiana Lafayette": "Louisiana",
    "Loyola Chicago": "Loyola-Chicago", "McNeese St.": "McNeese St",
    "Michigan St.": "Michigan St", "Middle Tennessee": "MTSU",
    "Milwaukee": "WI Milwaukee", "Mississippi St.": "Mississippi St",
    "Mississippi Valley St.": "MS Valley St", "Montana St.": "Montana St",
    "Morehead St.": "Morehead St", "Morgan St.": "Morgan St",
    "Mount St. Mary's": "Mt St Mary's", "Murray St.": "Murray St",
    "Nebraska Omaha": "NE Omaha", "New Mexico St.": "New Mexico St",
    "Norfolk St.": "Norfolk St", "North Carolina A&T": "NC A&T",
    "North Carolina Central": "NC Central", "North Carolina St.": "NC State",
    "North Dakota St.": "N Dakota St", "Northern Colorado": "N Colorado",
    "Northern Kentucky": "N Kentucky", "Northwestern St.": "Northwestern LA",
    "Ohio St.": "Ohio St", "Oklahoma St.": "Oklahoma St",
    "Oregon St.": "Oregon St", "Penn St.": "Penn St",
    "Portland St.": "Portland St", "Prairie View A&M": "Prairie View",
    "SIU Edwardsville": "SIUE", "Saint Francis": "St Francis PA",
    "Saint Joseph's": "St Joseph's PA", "Saint Louis": "St Louis",
    "Saint Mary's": "St Mary's CA", "Saint Peter's": "St Peter's",
    "Sam Houston St.": "Sam Houston St", "San Diego St.": "San Diego St",
    "South Dakota St.": "S Dakota St", "Southeast Missouri St.": "SE Missouri St",
    "Southern": "Southern Univ", "St. Bonaventure": "St Bonaventure",
    "St. John's": "St John's", "Stephen F. Austin": "SF Austin",
    "Texas A&M Corpus Chris": "TAM C. Christi", "Texas Southern": "TX Southern",
    "UTSA": "UT San Antonio", "Utah St.": "Utah St",
    "Washington St.": "Washington St", "Weber St.": "Weber St",
    "Western Kentucky": "WKU", "Western Michigan": "W Michigan",
    "Wichita St.": "Wichita St", "Wright St.": "Wright St",
    "Illinois St.": "Illinois St", "Missouri St.": "Missouri St",
    "Nicholls St.": "Nicholls St", "South Carolina St.": "S Carolina St",
    "Delaware St.": "Delaware St", "Idaho St.": "Idaho St",
    "Jackson St.": "Jackson St", "Tennessee St.": "Tennessee St",
    "Youngstown St.": "Youngstown St", "Chicago St.": "Chicago St",
    "Tarleton St.": "Tarleton St", "Sacramento St.": "CS Sacramento",
    "Southern Illinois": "S Illinois", "Texas St.": "Texas St",
    "Eastern Michigan": "E Michigan", "Eastern Illinois": "E Illinois",
    "Central Michigan": "C Michigan", "Northern Illinois": "N Illinois",
    "Western Illinois": "W Illinois", "Southeast Louisiana": "SE Louisiana",
    "Loyola Marymount": "Loy Marymount", "San Jose St.": "San Jose St",
    "Southern Utah": "Southern Utah", "Central Arkansas": "Cent Arkansas",
    "Houston Christian": "Houston Chr", "Central Connecticut": "Central Conn",
    "Bethune Cookman": "Bethune-Cookman", "Bethune-Cookman": "Bethune-Cookman",
    "Charleston": "Col Charleston", "Detroit Mercy": "Detroit", "Detroit": "Detroit",
    "FIU": "FL International", "Georgia Southern": "Ga Southern",
    "Houston Baptist": "Houston Chr", "LIU Brooklyn": "LIU", "LIU": "LIU",
    "Loyola Maryland": "Loyola MD", "Loyola (MD)": "Loyola MD",
    "UMass": "Massachusetts", "UMass Lowell": "MA Lowell",
    "Maryland Eastern Shore": "MD E Shore", "Miami (FL)": "Miami FL",
    "Miami FL": "Miami FL", "Miami (OH)": "Miami OH", "Miami OH": "Miami OH",
    "North Carolina Asheville": "NC Asheville",
    "North Carolina Greensboro": "NC Greensboro",
    "North Carolina Wilmington": "NC Wilmington",
    "Penn": "Pennsylvania", "Purdue Fort Wayne": "PFW", "IPFW": "PFW",
    "South Carolina Upstate": "USC Upstate", "Southern Miss": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
    "St. Francis (PA)": "St Francis PA", "St. Francis Brooklyn": "St Francis NY",
    "St. Joseph's": "St Joseph's PA", "St. Mary's (CA)": "St Mary's CA",
    "St. Peter's": "St Peter's", "Tennessee Martin": "UT Martin",
    "UT Martin": "UT Martin", "Texas A&M Corpus Christi": "TAM C. Christi",
    "Texas Arlington": "UT Arlington", "UT Arlington": "UT Arlington",
    "Texas Rio Grande Valley": "UT Rio Grande", "UTRGV": "UT Rio Grande",
    "VCU": "VA Commonwealth", "Virginia Commonwealth": "VA Commonwealth",
    "VMI": "VMI", "Omaha": "NE Omaha",
}


def normalize_name(name: str) -> str:
    """Normalize external team name to Kaggle convention."""
    name = str(name).strip()
    if name in EXTERNAL_TO_KAGGLE:
        return EXTERNAL_TO_KAGGLE[name]
    return re.sub(r'\.(?=\s|$)', '', name)


# ── Loaders for individual KenPom/Barttorvik files (name-based) ─

def load_kenpom_efficiency(ext_dir: Path, name_to_id: dict) -> pd.DataFrame:
    """Load KenPom Efficiency CSV (individual file, requires name mapping).

    Returns DataFrame with: Season, TeamID, AdjOE, AdjDE, AdjEM, AdjTempo.
    """
    path = ext_dir / "kenpom" / "INT _ KenPom _ Efficiency.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["KaggleName"] = df["Team"].apply(normalize_name)
    df["TeamID"] = df["KaggleName"].map(name_to_id)

    out = df[["Season", "TeamID"]].copy()
    for c in df.columns:
        cl = c.lower().strip()
        if "adjusted offensive" in cl:
            out["AdjOE"] = df[c]
        elif "adjusted defensive" in cl:
            out["AdjDE"] = df[c]
        elif "adjusted tem" in cl and "rank" not in cl:
            out["AdjTempo"] = df[c]

    if "AdjOE" in out.columns and "AdjDE" in out.columns:
        out["AdjEM"] = out["AdjOE"] - out["AdjDE"]

    return out.dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


def load_barttorvik_individual(ext_dir: Path, name_to_id: dict) -> pd.DataFrame:
    """Load Barttorvik yearly CSVs (requires name mapping).

    Returns DataFrame with: YEAR, TeamID, BARTHAG, ADJOE, ADJDE, EFG_O, EFG_D, WAB, ADJ_T.
    """
    files = []
    for yr in range(13, 26):
        f = ext_dir / "barttorvik" / f"cbb{yr}.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip().upper() for c in df.columns]
            if "YEAR" not in df.columns:
                df["YEAR"] = 2000 + yr
            files.append(df)

    combined_path = ext_dir / "barttorvik" / "cbb.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
        df.columns = [c.strip().upper() for c in df.columns]
        files.append(df)

    if not files:
        return pd.DataFrame()

    bart = pd.concat(files, ignore_index=True).drop_duplicates(
        subset=["TEAM", "YEAR"], keep="last"
    )
    bart["KaggleName"] = bart["TEAM"].apply(normalize_name)
    bart["TeamID"] = bart["KaggleName"].map(name_to_id)

    keep_cols = ["YEAR", "TeamID"]
    for src in ["BARTHAG", "ADJOE", "ADJDE", "EFG_O", "EFG_D", "WAB", "ADJ_T"]:
        if src in bart.columns:
            keep_cols.append(src)

    return bart[keep_cols].dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


# ── Loaders for multisource files (TEAM NO = direct TeamID) ─────

def _load_multisource(ext_dir: Path, filename: str) -> pd.DataFrame | None:
    """Helper to load a multisource CSV with TEAM NO -> TeamID mapping."""
    path = ext_dir / "multisource" / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "TEAM NO" in df.columns:
        df = df.rename(columns={"TEAM NO": "TeamID"})
    if "YEAR" in df.columns:
        df = df.rename(columns={"YEAR": "Season"})
    return df


def load_kp_bt_combined(ext_dir: Path) -> pd.DataFrame:
    """Load KenPom+Barttorvik combined (103 cols, 2008-2025, TEAM NO direct).

    Selects the most useful non-rank features. Returns DataFrame with Season, TeamID, features.
    """
    df = _load_multisource(ext_dir, "KenPom Barttorvik.csv")
    if df is None:
        return pd.DataFrame()

    # Select useful features (skip ranks, keep raw values)
    feature_cols = [
        "KADJ T", "KADJ O", "KADJ D", "KADJ EM",  # KenPom adjusted
        "BADJ EM", "BADJ O", "BADJ D", "BARTHAG",  # Barttorvik adjusted
        "EFG%", "EFG%D", "FTR", "FTRD",             # Four factors
        "TOV%", "TOV%D", "OREB%", "DREB%",
        "2PT%", "2PT%D", "3PT%", "3PT%D",            # Shooting
        "BLK%", "BLKED%", "AST%",                    # Advanced
        "AVG HGT", "EFF HGT", "EXP", "TALENT",      # Team composition
        "FT%", "PPPO", "PPPD", "ELITE SOS", "WAB",   # Misc
    ]

    keep = ["Season", "TeamID"]
    for c in feature_cols:
        if c in df.columns:
            # Clean column name for use as feature
            clean = "KPBT_" + c.replace(" ", "_").replace("%", "pct")
            df[clean] = pd.to_numeric(df[c], errors="coerce")
            keep.append(clean)

    return df[keep].dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


def load_resumes(ext_dir: Path) -> pd.DataFrame:
    """Load Resumes data (Q1/Q2 wins, SOS metrics, 2008-2025, TEAM NO direct).

    Returns DataFrame with Season, TeamID, resume features.
    """
    df = _load_multisource(ext_dir, "Resumes.csv")
    if df is None:
        return pd.DataFrame()

    feature_cols = [
        "NET RPI", "RESUME", "WAB RANK", "ELO", "B POWER",
        "Q1 W", "Q2 W", "Q1 PLUS Q2 W", "Q3 Q4 L", "PLUS 500", "R SCORE",
    ]

    keep = ["Season", "TeamID"]
    for c in feature_cols:
        if c in df.columns:
            clean = "RES_" + c.replace(" ", "_")
            df[clean] = pd.to_numeric(df[c], errors="coerce")
            keep.append(clean)

    return df[keep].dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


def load_evan_miya(ext_dir: Path) -> pd.DataFrame:
    """Load EvanMiya ratings (2013-2025, TEAM NO direct).

    Returns DataFrame with Season, TeamID, EvanMiya features.
    """
    df = _load_multisource(ext_dir, "EvanMiya.csv")
    if df is None:
        return pd.DataFrame()

    feature_cols = [
        "O RATE", "D RATE", "RELATIVE RATING",
        "OPPONENT ADJUST", "PACE ADJUST", "TRUE TEMPO",
        "KILL SHOTS PER GAME", "KILL SHOTS CONCEDED PER GAME",
    ]

    keep = ["Season", "TeamID"]
    for c in feature_cols:
        if c in df.columns:
            clean = "EM_" + c.replace(" ", "_")
            df[clean] = pd.to_numeric(df[c], errors="coerce")
            keep.append(clean)

    return df[keep].dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


def load_538_ratings(ext_dir: Path) -> pd.DataFrame:
    """Load FiveThirtyEight power ratings (2016-2024, TEAM NO direct).

    Returns DataFrame with Season, TeamID, 538 features.
    """
    df = _load_multisource(ext_dir, "538 Ratings.csv")
    if df is None:
        return pd.DataFrame()

    keep = ["Season", "TeamID"]
    if "POWER RATING" in df.columns:
        df["F538_POWER_RATING"] = pd.to_numeric(df["POWER RATING"], errors="coerce")
        keep.append("F538_POWER_RATING")

    return df[keep].dropna(subset=["TeamID"]).copy().astype({"TeamID": int})


def load_coach_features(ext_dir: Path, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """Load coach tournament features and merge with Kaggle MTeamCoaches.

    coaches_df: MTeamCoaches.csv DataFrame with Season, TeamID, CoachName columns.
    Returns DataFrame with Season, TeamID, coach features (PAKE, PASE, F4%).
    """
    df = _load_multisource(ext_dir, "Coach Results.csv")
    if df is None or coaches_df is None:
        return pd.DataFrame()

    # Coach Results has COACH name but no season/team. Merge via MTeamCoaches.
    # Normalize coach names for matching
    df["coach_key"] = df["COACH"].str.strip().str.lower()
    coaches = coaches_df.copy()
    coaches["coach_key"] = coaches["CoachName"].str.strip().str.lower()

    merged = coaches.merge(df[["coach_key", "PAKE", "PASE", "F4%", "WIN%", "GAMES"]],
                           on="coach_key", how="left")

    # Parse F4% (string like "95.50%")
    if "F4%" in merged.columns:
        merged["COACH_F4pct"] = pd.to_numeric(
            merged["F4%"].astype(str).str.replace("%", ""), errors="coerce"
        )
    merged["COACH_PAKE"] = pd.to_numeric(merged.get("PAKE"), errors="coerce")
    merged["COACH_PASE"] = pd.to_numeric(merged.get("PASE"), errors="coerce")
    merged["COACH_GAMES"] = pd.to_numeric(merged.get("GAMES"), errors="coerce")

    keep = ["Season", "TeamID", "COACH_PAKE", "COACH_PASE", "COACH_F4pct", "COACH_GAMES"]
    return merged[[c for c in keep if c in merged.columns]].copy()
