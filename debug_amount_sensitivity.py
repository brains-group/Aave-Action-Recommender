from actionAgentTraining import (
    get_date_ranges,
    get_model_for_pair_and_date,
    preprocess,
    verify_amount_feature_effect,
    get_event_df,
)
import pandas as pd

# pick a sample: load some event df, choose a row with large amounts
df = get_event_df('borrow','liquidated')  # adjust pair if needed
row = df.sample(n=1, random_state=42).iloc[0]

train_dates, test_dates = get_date_ranges()
dates = train_dates.union(test_dates)
model_date = dates[dates <= pd.to_datetime(row['timestamp'], unit='s')].max()
index_event_title = str(row['Index Event']).title()

# ensure Outcome Event present for preprocess
row_copy = row.copy()
row_copy['Outcome Event'] = 'liquidated'

_, _, test_feats = preprocess(test_features_df=row_copy.to_frame().T, model_date=model_date)
model = get_model_for_pair_and_date(index_event_title, 'Liquidated', model_date=model_date, verbose=True)

print('test_feats.shape', None if test_feats is None else test_feats.shape)
print('columns sample:', None if test_feats is None else test_feats.columns[:50].tolist())

res = verify_amount_feature_effect(model, test_feats, index_event=index_event_title, outcome_event='Liquidated', model_date=model_date)
import json, pprint
pprint.pprint(res)
