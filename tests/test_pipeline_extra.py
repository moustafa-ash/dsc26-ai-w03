import numpy as np

from tawseel.data import make_parcels
from tawseel.model import add_history_features, build_pipeline, time_split


def test_pipeline_handles_new_and_missing_values():
    df = add_history_features(make_parcels(500, seed=7))
    train, test = time_split(df)
    model = build_pipeline(seed=0)
    model.fit(train, train["failed"])

    examples = test.iloc[:2].copy()
    examples.loc[examples.index[0], "area"] = "New unseen area"
    examples.loc[examples.index[0], "merchant_id"] = "New merchant"
    examples.loc[examples.index[0], "promised_slot"] = None
    examples.loc[examples.index[0], "address_text"] = ""
    examples.loc[
        examples.index[0],
        [
            "cod_amount",
            "n_items",
            "customer_prior_parcels",
            "customer_prior_fail_rate",
        ],
    ] = np.nan

    batch_probability = model.predict_proba(examples)[0, 1]
    single_probability = model.predict_proba(examples.iloc[[0]])[0, 1]

    assert np.isfinite(batch_probability)
    assert 0 <= batch_probability <= 1
    assert np.isclose(batch_probability, single_probability)
