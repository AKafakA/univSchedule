import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch
import pytorch_lightning as pl

from predictor.instance_predictor.latency_model import LSTMLatencyModel
from predictor.data_process.data_module import RequestTraceDataModule

# Load the data
# dataset_path = "predictor/test/data/dummy_res_trace.csv"
# dataset = RequestTraceDataset(filepath=dataset_path, lookback_steps=1)
# dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=request_trace_collate_fn)
#
# # Initialize the model
# model = LSTMLatencyModel()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Train the model
# for epoch in range(10):
#     running_loss = 0.0
#     for i, (X_batch, y_batch) in enumerate(dataloader):
#         optimizer.zero_grad()
#         output = model(X_batch)
#         loss = criterion(output, y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

# Load the data
datamodule = RequestTraceDataModule(
    data_path="predictor/test/data/dummy_res_trace.csv",
    lookback_steps=1,
    batch_size=1,
    num_workers=0
)
model = LSTMLatencyModel()
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, datamodule)

