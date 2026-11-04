# AI usage

<!-- Replace every comment below with your own words. Short is fine. The grader checks only
     that this file exists and has been changed from this template; your Track Mentor reads it. -->

## Tools I used, and for what

I used Codex to explain the task, guide me through the implementation,
and suggest tests for history, the time split, and the pipeline.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I rejected a suggestion to use a row-by-row cumulative count for customer
history. I tested it with two parcels at the same timestamp: it returned
[0, 1, 2] instead of [0, 1, 1], so it incorrectly counted one tied parcel
as prior to the other.

## How I checked that my model has no leak

I excluded post-delivery columns and the target from the model features,
checked that customer history uses only earlier timestamps, and split
training and test data by time.
