import pandas as pd
from torch.utils.data import DataLoader

from predictor.data_process.dataset import RequestTraceDataset, request_trace_collate_fn

dataframe = pd.read_csv('predictor/test/data/dummy_res_trace.csv')
datasets = RequestTraceDataset(dataframe=dataframe, lookback_steps=1)
dataloader = DataLoader(datasets, batch_size=1, shuffle=False, collate_fn=request_trace_collate_fn)

for i, (X_batch, y_batch) in enumerate(dataloader):
    print(X_batch, y_batch)
