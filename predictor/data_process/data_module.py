import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from predictor.data_process.dataset import RequestTraceDataset, request_trace_collate_fn


class RequestTraceDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path,
                 lookback_steps=1,
                 shuffle_training=False,
                 train_size=0.8,
                 val_size=0.1,
                 seq_len=1,
                 batch_size=128,
                 num_workers=23):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.shuffle_training = shuffle_training
        self.train_size = train_size
        self.val_size = val_size
        self.lookback_steps = lookback_steps

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self, stage):
        """
        * read Data
        * 'Date' and 'Time' columns are merged into 'date' index
        * convert all to float and delete nans
        * resampled to hourly intervals
        * define X (features) and y (lables)
        """
        # read data
        df = pd.read_csv(self.data_path)

        # change types to float (and all no number values to nan)
        for i in range(len(df.columns)):
            df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors='coerce')

        # delete nans
        df = df.dropna()

        # split data into train, val and test
        train_df, test_df = train_test_split(df, train_size=self.train_size, shuffle=self.shuffle_training)
        val_ratio = self.val_size / (1 - self.train_size)
        val_df, test_df = train_test_split(test_df, train_size=val_ratio, shuffle=self.shuffle_training)

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def train_dataloader(self):
        """
        * no further transformation necessary
        * wrap dataset in dataloader
        """
        # create dataset
        train_dataset = RequestTraceDataset(self.train_df, lookback_steps=self.lookback_steps)

        # wrap dataset in dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_training,
                                      num_workers=self.num_workers, collate_fn=request_trace_collate_fn)

        return train_dataloader

    def val_dataloader(self):
        # create dataset
        val_dataset = RequestTraceDataset(self.val_df, lookback_steps=self.lookback_steps)

        # wrap dataset in dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_training,
                                    num_workers=self.num_workers, collate_fn=request_trace_collate_fn)

        return val_dataloader

    def test_dataloader(self):
        # create dataset
        test_dataset = RequestTraceDataset(self.test_df, lookback_steps=self.lookback_steps)

        # wrap dataset in dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle_training,
                                     num_workers=self.num_workers, collate_fn=request_trace_collate_fn)

        return test_dataloader
