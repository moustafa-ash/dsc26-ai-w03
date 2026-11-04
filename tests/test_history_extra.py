import pandas as pd
import pandas.testing as pdt

from tawseel.model import add_history_features


def test_ties_and_shuffled_rows():
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C1"],
            "created_at": pd.to_datetime(
                [
                    "2026-06-01 09:00",
                    "2026-06-01 10:00",
                    "2026-06-01 10:00",
                    "2026-06-01 11:00",
                ]
            ),
            "failed": [1, 0, 1, 0],
        },
        index=[10, 20, 30, 40],
    )

    original = df.copy(deep=True)
    result = add_history_features(df)
    shuffled = add_history_features(df.sample(frac=1, random_state=42))

    assert result["customer_prior_parcels"].tolist() == [0, 1, 1, 3]
    assert result.loc[20, "customer_prior_fail_rate"] == 1.0
    assert result.loc[30, "customer_prior_fail_rate"] == 1.0
    assert result.loc[40, "customer_prior_fail_rate"] == 2 / 3
    pdt.assert_frame_equal(df, original)
    pdt.assert_frame_equal(result.sort_index(), shuffled.sort_index())
