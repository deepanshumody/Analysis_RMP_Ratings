import numpy as np

from ape import config, data


def test_config_constants():
    assert config.RANDOM_SEED == 10676128
    assert config.ALPHA == 0.005
    assert config.MIN_RATINGS == 10
    assert len(config.NUM_COLS) == 8
    assert len(config.TAG_COLS) == 20
    assert len(config.QUAL_COLS) == 3
    assert (config.DATA_DIR / "rmpCapstoneNum.csv").exists()
    assert set(config.TAG_LABELS) == set(config.TAG_COLS)


def test_load_numeric_shape_and_columns():
    df = data.load_numeric()
    assert df.shape == (89893, 8)
    assert list(df.columns) == config.NUM_COLS


def test_load_tags_shape():
    assert data.load_tags().shape == (89893, 20)


def test_filter_min_ratings_count():
    df = data.load_numeric()
    assert len(data.filter_min_ratings(df)) == 9841


def test_gender_subset_is_exclusive_and_counts():
    df = data.filter_min_ratings(data.load_numeric())
    g = data.gender_subset(df)
    assert len(g) == 7105
    assert ((g["high_conf_male"] + g["high_conf_female"]) == 1).all()


def test_normalize_tags_rows_sum_to_one():
    tags = data.load_tags().iloc[:300].astype(float)
    denom = tags.sum(axis=1)
    norm = data.normalize_tags(tags, denom=denom)
    # rows with at least one tag should sum to 1 after normalization;
    # rows with zero tags are 0/0 and are excluded
    row_sums = norm[denom > 0].sum(axis=1)
    assert np.allclose(row_sums, 1.0)
