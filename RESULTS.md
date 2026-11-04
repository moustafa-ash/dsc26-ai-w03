# Results: Tawseel failed-delivery baseline

<!-- Read by your Track Mentor, not graded by tests. Replace the comments. -->

## Baseline table

Last 20% of parcels by time, using `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
| --- | ---: | ---: | ---: | ---: |
| Dummy (prior) | 0.500 | 0.220 | 0.172 | 0.220 |
| My model | 0.630 | 0.342 | 0.166 | 0.220 |

## Three sentences for Sherif

The model helps rank parcels by their estimated chance of delivery failure.
On our synthetic test data, it performed better than predicting the same risk for every parcel.
We still need to test it on real delivery data before relying on it for operational decisions.
