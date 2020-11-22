"""Train a model from a training dataset."""

import argparse
import math

import pl_utils as plu

import pytorch_lightning as lt

import torch


class VectorDataset(torch.utils.data.Dataset):
    """Vector Dataset class to facilitate training."""

    def __init__(self, input_data, output_data):
        """Initialize input and output dataset."""
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        """Compute length of dataset."""
        return len(self.input_data)

    def __getitem__(self, idx):
        """Recover an item of dataset."""
        return self.input_data[idx], self.output_data[idx]


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class RandomDataModule(lt.core.datamodule.LightningDataModule):
    """Data module to load train/val/test dataloaders."""

    def __init__(self, data):
        """Initialze variables."""
        super().__init__()
        self.data = data
        self.batch_size = 100
        self.num_workers = 4

        self.train_input_data = None
        self.train_output_data = None
        self.val_input_data = None
        self.val_output_data = None
        self.test_input_data = None
        self.test_output_data = None

    def setup(self, stage=None):
        """Create and assign splits."""
        train_pct = 0.8
        val_pct = 0.1

        samples = len(self.data)

        train_samples_idx = math.floor(train_pct * samples)
        val_samples_idx = train_samples_idx + math.floor(val_pct * samples)

        train_data = self.data[0:train_samples_idx]
        val_data = self.data[train_samples_idx:val_samples_idx]
        test_data = self.data[val_samples_idx:None]

        self.train_input_data, self.train_output_data = \
            self._split_input_output_data(train_data)
        self.val_input_data, self.val_output_data = \
            self._split_input_output_data(val_data)
        self.test_input_data, self.test_output_data = \
            self._split_input_output_data(test_data)

    def train_dataloader(self, *args, **kwargs):
        """Create train dataloader."""
        train_split = VectorDataset(self.train_input_data,
                                    self.train_output_data)
        return torch.utils.data.DataLoader(dataset=train_split,
                                           num_workers=self.num_workers,
                                           batch_size=self.batch_size,
                                           shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        """Create val dataloader."""
        val_split = VectorDataset(self.val_input_data, self.val_output_data)
        return torch.utils.data.DataLoader(dataset=val_split,
                                           num_workers=self.num_workers,
                                           batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs):
        """Create test dataloader."""
        test_split = VectorDataset(self.test_input_data, self.test_output_data)
        return torch.utils.data.DataLoader(dataset=test_split,
                                           num_workers=self.num_workers,
                                           batch_size=self.batch_size)

    @staticmethod
    def _split_input_output_data(data):
        input_dim = 3
        input_data = data[:, 0:input_dim]
        output_data = data[:, input_dim:None]
        return input_data, output_data


def main():
    """Initialize model and trainer to fit a fc net."""
    # Create random data to fit a fully connected network
    samples = 10000
    input_dim = 3
    output_dim = 4
    random_data = torch.rand(samples, input_dim + output_dim)

    # Construct lightning data module for the dataset
    data_module = RandomDataModule(random_data)

    parser = argparse.ArgumentParser()

    # Add trainer specific args from model
    parser = lt.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.fc.FCModel.add_model_specific_args(parser)

    training_args_list = ['--auto_lr_find', 'False',
                          '--auto_scale_batch_size', 'False',
                          '--benchmark', 'True',
                          '--fast_dev_run', 'False',
                          '--gpus', '-1',
                          '--precision', '16',
                          '--terminate_on_nan', 'True',
                          '--weights_summary', 'full']

    model_args_list = ['--layer_dims', '3 32 32 4',
                       '--learning_rate', '0.003981071705534969']

    args_list = training_args_list + model_args_list

    hparams = parser.parse_args(args_list)

    # create model
    model = plu.models.fc.FCModel(hparams)

    # create trainer
    trainer = lt.Trainer.from_argparse_args(hparams)

    # tune trainer
    trainer.tune(model, datamodule=data_module)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
