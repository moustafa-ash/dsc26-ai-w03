import pandas as pd
import pytest

from tawseel.model import time_split


def test_boundary_ties_stay_in_test():
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(
                [
                    "2026-06-01",
                    "2026-06-02",
                    "2026-06-03",
                    "2026-06-03",
                    "2026-06-04",
                ]
            ),
        },
        index=[10, 20, 30, 40, 50],
    )

    train, test = time_split(df, test_fraction=0.4)

    assert train.index.tolist() == [10, 20]
    assert set(test.index) == {30, 40, 50}


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_invalid_fraction_raises(fraction):
    df = pd.DataFrame({"created_at": pd.to_datetime(["2026-06-01"])})

    with pytest.raises(ValueError):
        time_split(df, test_fraction=fraction)
