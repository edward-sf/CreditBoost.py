import pytest

from creditboost import config
from creditboost.data import MissingColumnsError, file_sha256, load_training_frame, split


def test_loads_the_fixture(fixture_path):
    frame = load_training_frame(fixture_path)
    assert len(frame) == 200
    assert config.TARGET_COLUMN in frame.columns


def test_missing_required_column_raises_and_names_it(fixture_path, tmp_path):
    frame = load_training_frame(fixture_path).drop(columns=["AMT_CREDIT"])
    truncated = tmp_path / "truncated.csv"
    frame.to_csv(truncated, index=False)
    with pytest.raises(MissingColumnsError, match="AMT_CREDIT"):
        load_training_frame(truncated)


def test_absent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_frame(tmp_path / "nope.csv")


def test_sha256_is_stable_and_content_sensitive(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("hello")
    second.write_text("hello")
    assert file_sha256(first) == file_sha256(second)
    second.write_text("goodbye")
    assert file_sha256(first) != file_sha256(second)


def test_split_partitions_every_row_exactly_once(fixture_path):
    frame = load_training_frame(fixture_path)
    train, valid = split(frame)
    assert len(train) + len(valid) == len(frame)
    assert set(train.index).isdisjoint(valid.index)


def test_split_is_stratified_on_the_target(fixture_path):
    """At an 8% base rate an unstratified split can miss positives entirely."""
    frame = load_training_frame(fixture_path)
    train, valid = split(frame)
    overall = frame[config.TARGET_COLUMN].mean()
    assert train[config.TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.03)
    assert valid[config.TARGET_COLUMN].mean() == pytest.approx(overall, abs=0.03)


def test_split_is_reproducible_for_a_fixed_seed(fixture_path):
    frame = load_training_frame(fixture_path)
    first, _ = split(frame)
    second, _ = split(frame)
    assert list(first.index) == list(second.index)
