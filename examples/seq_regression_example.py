"""Train a model from a training dataset."""

import argparse
import math

import pl_utils as plu

import pytorch_lightning as lt

import torch


class VectorSeqDataset(torch.utils.data.Dataset):
    """Vector Seq Dataset class to facilitate training."""

    def __init__(self, input_data, output_data):
        """Initialize input and output dataset."""
        self.input_data = input_data[0]
        hidden_data = input_data[1]
        self.hidden_state_data = hidden_data[0]
        self.cell_state_data = hidden_data[1]
        self.output_data = output_data

    def __len__(self):
        """Compute length of dataset."""
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        """Recover an item of dataset."""
        return self.input_data[idx], \
            (self.hidden_state_data[idx], self.cell_state_data[idx]), \
            self.output_data[idx]


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class RandomSeqDataModule(lt.core.datamodule.LightningDataModule):
    """Data module to load train/val/test dataloaders."""

    def __init__(self, hparams, data):
        """Initialze variables."""
        super().__init__()
        self.hparams = hparams

        self.data = data

        self.train_input_data = None
        self.train_output_data = None
        self.val_input_data = None
        self.val_output_data = None
        self.test_input_data = None
        self.test_output_data = None

    # pylint: disable=too-many-locals
    def setup(self, stage=None):
        """Create and assign splits."""
        train_pct = 0.8
        val_pct = 0.1

        input_output_data = self.data[0]
        initial_hidden_data = self.data[1]
        samples = input_output_data.shape[0]

        train_samples_idx = math.floor(train_pct * samples)
        val_samples_idx = train_samples_idx + math.floor(val_pct * samples)

        train_data = (input_output_data[0:train_samples_idx],
                      initial_hidden_data[0:train_samples_idx])
        val_data = (input_output_data[train_samples_idx:val_samples_idx],
                    initial_hidden_data[train_samples_idx:val_samples_idx])
        test_data = (input_output_data[val_samples_idx:None],
                     initial_hidden_data[val_samples_idx:None])

        train_input_data, train_output_data = \
            self._split_input_output_data(train_data[0])
        val_input_data, val_output_data = \
            self._split_input_output_data(val_data[0])
        test_input_data, test_output_data = \
            self._split_input_output_data(test_data[0])

        train_initial_hidden_data, train_initial_cell_data = \
            self._split_hidden_cell_data(train_data[1])
        val_initial_hidden_data, val_initial_cell_data = \
            self._split_hidden_cell_data(val_data[1])
        test_initial_hidden_data, test_initial_cell_data = \
            self._split_hidden_cell_data(test_data[1])

        self.train_input_data = (train_input_data, (train_initial_hidden_data,
                                                    train_initial_cell_data))
        self.train_output_data = train_output_data

        self.val_input_data = (val_input_data, (val_initial_hidden_data,
                                                val_initial_cell_data))
        self.val_output_data = val_output_data

        self.test_input_data = (test_input_data, (test_initial_hidden_data,
                                                  test_initial_cell_data))
        self.test_output_data = test_output_data

    def train_dataloader(self, *args, **kwargs):
        """Create train dataloader."""
        train_split = VectorSeqDataset(self.train_input_data,
                                       self.train_output_data)
        return torch.utils.data.DataLoader(
            dataset=train_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        """Create val dataloader."""
        val_split = VectorSeqDataset(self.val_input_data, self.val_output_data)
        return torch.utils.data.DataLoader(
            dataset=val_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size)

    def test_dataloader(self, *args, **kwargs):
        """Create test dataloader."""
        test_split = VectorSeqDataset(self.test_input_data,
                                      self.test_output_data)
        return torch.utils.data.DataLoader(
            dataset=test_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size)

    @staticmethod
    def _split_hidden_cell_data(data):
        hidden_state_dim = data.shape[2] // 2
        hidden_state_data = data[:, :, 0:hidden_state_dim]
        cell_state_data = data[:, :, hidden_state_dim:None]
        return hidden_state_data, cell_state_data

    @staticmethod
    def _split_input_output_data(data):
        input_dim = 3
        input_data = data[:, :, 0:input_dim]
        output_data = data[:, :, input_dim:None]
        return input_data, output_data


def main():
    """Initialize model and trainer to fit a lstm net."""
    parser = argparse.ArgumentParser()

    # Add program specific args from model
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--data_num_workers', type=int, default=4)

    # Add trainer specific args from model
    parser = lt.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.lstm.LSTMModel.add_model_specific_args(parser)

    program_args_list = ['--batch_size', '2',
                         '--data_num_workers', '4']

    training_args_list = ['--accumulate_grad_batches', '1',
                          '--auto_lr_find', 'False',
                          '--auto_scale_batch_size', 'False',
                          '--benchmark', 'True',
                          '--fast_dev_run', '0',
                          '--gpus', '-1',
                          '--precision', '16',
                          '--terminate_on_nan', 'True',
                          '--weights_summary', 'full']

    model_args_list = ['--input_size', '3',
                       '--hidden_size', '4',
                       '--num_layers', '3',
                       '--learning_rate', '0.002']

    args_list = program_args_list + training_args_list + model_args_list

    hparams_args = parser.parse_args(args_list)
    hparams = vars(hparams_args)

    # Create random data to fit a fully connected network
    sequence_dim = 5
    samples = 20
    input_dim = 3
    hidden_dim = 4
    num_layers = 3
    # Want to change this make samples up front
    random_seq_data = (torch.rand(samples, sequence_dim,
                                  input_dim + hidden_dim),
                       torch.rand(samples, num_layers, hidden_dim * 2))

    # Construct lightning data module for the dataset
    data_module = RandomSeqDataModule(hparams_args, random_seq_data)

    # create model
    model = plu.models.lstm.LSTMModel(**hparams)

    # create trainer
    trainer = lt.Trainer.from_argparse_args(hparams_args)

    # tune trainer
    trainer.tune(model, datamodule=data_module)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
