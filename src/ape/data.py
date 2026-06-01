"""Load and clean the RateMyProfessor capstone data."""

from __future__ import annotations

import pandas as pd

from . import config


def load_numeric(path=None) -> pd.DataFrame:
    """Load rmpCapstoneNum.csv with canonical column names."""
    path = path or config.DATA_DIR / "rmpCapstoneNum.csv"
    df = pd.read_csv(path, header=None)
    df.columns = config.NUM_COLS
    return df


def load_tags(path=None) -> pd.DataFrame:
    """Load rmpCapstoneTags.csv with canonical column names."""
    path = path or config.DATA_DIR / "rmpCapstoneTags.csv"
    df = pd.read_csv(path, header=None)
    df.columns = config.TAG_COLS
    return df


def load_qual(path=None) -> pd.DataFrame:
    """Load rmpCapstoneQual.csv with canonical column names."""
    path = path or config.DATA_DIR / "rmpCapstoneQual.csv"
    df = pd.read_csv(path, header=None)
    df.columns = config.QUAL_COLS
    return df


def filter_min_ratings(df: pd.DataFrame, k: int = config.MIN_RATINGS) -> pd.DataFrame:
    """Keep professors whose average is based on at least ``k`` ratings."""
    return df[df["num_ratings"] >= k].copy()


def gender_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows confidently labelled exactly one gender (male XOR female)."""
    mask = (df["high_conf_male"] + df["high_conf_female"]) == 1
    return df[mask].copy()


def normalize_tags(tags: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    """Normalize raw tag counts by a per-row denominator (tag total or #ratings)."""
    return tags.div(denom, axis=0)
