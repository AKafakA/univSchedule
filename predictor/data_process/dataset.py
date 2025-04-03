import torch
from torch.utils.data import Dataset


def _load_data(dataframe):
    data = []
    for index, row in dataframe.iterrows():
        data.append({
            'request_id': row['id'],
            'timestamp': row['timestamp'],
            'latency': row['latency'],
            'input_length': row['input'],
            'output_length': row['output']
        })
    data.sort(key=lambda x: x['timestamp'])
    return data


class RequestTraceDataset(Dataset):
    def __init__(self, dataframe, lookback_steps=3):
        self.lookback_steps = lookback_steps
        self.raw_data = _load_data(dataframe)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        current_request = self.raw_data[idx]
        timestamp = current_request['timestamp']
        first_unfinished_request = next((i for i, x in enumerate(self.raw_data)
                                         if x['timestamp'] + x['latency'] > timestamp), idx)
        first_index = max(first_unfinished_request - self.lookback_steps, 0)
        lookback_requests = []
        for i in range(first_index, idx):
            selected_row = self.raw_data[i]
            masked_latency = -1
            if selected_row['timestamp'] + selected_row['latency'] < timestamp:
                masked_latency = selected_row['latency']
            lookback_requests.append({
                'timestamp': selected_row['timestamp'],
                'latency': masked_latency,
                'input_length': selected_row['input_length'],
                'output_length': selected_row['output_length']
            })
        lookback_requests.append(
            {
                'timestamp': current_request['timestamp'],
                'latency': -1,
                'input_length': current_request['input_length'],
                'output_length': current_request['output_length']
            }
        )
        return lookback_requests, current_request['latency']


def request_trace_collate_fn(batch):
    X, y = zip(*batch)
    # print(X)
    # features = [torch.tensor([item[i] for item in batch], dtype=torch.float32) for i in range(1, 4)]
    features = []
    for i in range(len(X)):
        feature = []
        for j in range(0, len(X[i])):
            feature.append(list(X[i][j].values()))
        features.append(torch.tensor(feature, dtype=torch.float32))
    X = torch.stack(features, dim=0)
    return X, torch.tensor(y, dtype=torch.float32)
