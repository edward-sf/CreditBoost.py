import pytest

from creditboost.banding import risk_band


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "low"),
        (0.05, "low"),
        (0.0999, "low"),
        (0.10, "medium"),
        (0.20, "medium"),
        (0.2999, "medium"),
        (0.30, "high"),
        (0.95, "high"),
        (1.0, "high"),
    ],
)
def test_bands_are_assigned_at_the_configured_boundaries(probability, expected):
    assert risk_band(probability) == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_zero_to_one_raises(probability):
    with pytest.raises(ValueError):
        risk_band(probability)
