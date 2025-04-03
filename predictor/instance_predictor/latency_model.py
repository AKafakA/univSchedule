import torch.nn as nn
import torch
import pytorch_lightning as pl


class LSTMLatencyModel(pl.LightningModule):
    def __init__(self,
                 n_features=4,
                 hidden_size=50,
                 batch_size=1,
                 num_layers=2,
                 dropout=0.1,
                 learning_rate=0.001,
                 criterion=nn.MSELoss()):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.criterion = criterion

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_units).requires_grad_()
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[0]).flatten()
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        # result = pl.EvalResult(checkpoint_on=loss)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        # result = pl.EvalResult()
        self.log('test_loss', loss)
        return loss
