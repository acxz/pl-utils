"""Train a model from a training dataset."""

import argparse

import pl_utils as plu

import pytorch_lightning as pl

import torch


class VectorSeqDataset(torch.utils.data.Dataset):
    """Vector Seq Dataset class to facilitate training."""

    def __init__(self, input_sequence_data, input_hidden_cell_data,
                 output_sequence_data, output_hidden_cell_data):
        """Initialize dataset."""
        (input_hidden_data, input_cell_data) = input_hidden_cell_data
        (output_hidden_data, output_cell_data) = output_hidden_cell_data

        self.input_sequence_data = input_sequence_data
        self.input_hidden_data = input_hidden_data
        self.input_cell_data = input_cell_data
        self.output_sequence_data = output_sequence_data
        self.output_hidden_data = output_hidden_data
        self.output_cell_data = output_cell_data

    def __len__(self):
        """Compute length of dataset."""
        return len(self.input_sequence_data)

    def __getitem__(self, idx):
        """Recover an item of dataset."""
        return (self.input_sequence_data[idx], (self.input_hidden_data[idx],
                self.input_cell_data[idx]), self.output_sequence_data[idx],
                (self.output_hidden_data[idx], self.output_cell_data[idx]))


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class VectorSeqDataModule(pl.core.datamodule.LightningDataModule):
    """Data module to load train/val/test dataloaders."""

    def __init__(self, hparams, data):
        """Initialze variables."""
        super().__init__()
        self.save_hyperparameters(hparams)

        self.data = data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # pylint: disable=too-many-locals
    def setup(self, stage=None):
        """Create and assign splits."""
        train_pct = 0.8
        test_pct = 0.1

        samples = len(self.data)

        train_samples = int(train_pct * samples)
        test_samples = int(test_pct * samples)
        val_samples = samples - train_samples - test_samples

        input_sequence_data, (input_hidden_data, input_cell_data), \
            output_sequence_data, (output_hidden_data, output_cell_data) \
            = self._split_data(self.data)

        vector_seq_dataset = VectorSeqDataset(
            input_sequence_data, (input_hidden_data, input_cell_data),
            output_sequence_data, (output_hidden_data, output_cell_data))

        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                vector_seq_dataset, [train_samples, val_samples, test_samples])

    def train_dataloader(self, *args, **kwargs):
        """Create train dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        """Create val dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size)

    def test_dataloader(self, *args, **kwargs):
        """Create test dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.batch_size)

    @staticmethod
    def _split_data(data):
        sequence_dim = 5
        input_dim = 3
        hidden_dim = 4
        num_layers = 3
        bidirectional = False
        proj_size = 0

        if bidirectional:
            d = 2
        else:
            d = 1
        if proj_size > 0:
            h_out = proj_size
        else:
            h_out = hidden_dim

        input_sequence_data, input_hidden_data, input_cell_data, \
            output_sequence_data, output_hidden_data, output_cell_data \
            = torch.split(
                data,
                [input_dim * sequence_dim, h_out * d * num_layers,
                 hidden_dim * d * num_layers, h_out * d * sequence_dim,
                 h_out * d * num_layers, hidden_dim * d * num_layers],
                1)

        # Reshape input data
        # TODO: Maybe this should already be handled in input data
        samples = len(data)
        input_sequence_data = torch.reshape(
            input_sequence_data, (samples, sequence_dim, input_dim))
        input_hidden_data = torch.reshape(
            input_hidden_data, (samples, d * num_layers, h_out))
        input_cell_data = torch.reshape(
            input_cell_data, (samples, d * num_layers, hidden_dim))
        output_sequence_data = torch.reshape(
            output_sequence_data, (samples, sequence_dim, d * h_out))
        output_hidden_data = torch.reshape(
            output_hidden_data, (samples, d * num_layers, h_out))
        output_cell_data = torch.reshape(
            output_cell_data, (samples, d * num_layers, hidden_dim))

        return (input_sequence_data, (input_hidden_data, input_cell_data),
                output_sequence_data, (output_hidden_data, output_cell_data))


def main():
    """Initialize model and trainer to fit a lstm net."""
    parser = argparse.ArgumentParser()

    # Add program specific args from model
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_num_workers', type=int, default=4)

    # Add trainer specific args from model
    parser = pl.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.lstm.LSTMModel.add_model_specific_args(parser)

    program_args_list = ['--batch_size', '800',
                         '--data_num_workers', '4']

    training_args_list = ['--accelerator', 'auto',
                          '--accumulate_grad_batches', '1',
                          '--auto_lr_find', 'False',
                          '--auto_scale_batch_size', 'False',
                          '--benchmark', 'True',
                          '--enable_checkpointing', 'True',
                          '--detect_anomaly', 'True',
                          '--fast_dev_run', 'False',
                          '--enable_progress_bar', 'True',
                          '--precision', '16',
                          '--max_epochs', '10',
                          '--enable_model_summary', 'True']

    model_args_list = ['--input_size', '3',
                       '--hidden_size', '4',
                       '--num_layers', '3',
                       '--bias', 'True',
                       '--batch_first', 'True',
                       '--dropout', '0',
                       '--bidirectional', 'False',
                       '--proj_size', '0',
                       '--learning_rate', '0.002']

    args_list = program_args_list + training_args_list + model_args_list

    hparams_args = parser.parse_args(args_list)

    hparams = vars(hparams_args)

    # Create random data to fit a fully connected network
    # TODO: use more realistic input dimensions
    samples = 1000
    sequence_dim = 5
    input_dim = 3
    hidden_dim = 4
    num_layers = 3
    bidirectional = False
    proj_size = 0

    if bidirectional:
        d = 2
    else:
        d = 1
    if proj_size > 0:
        h_out = proj_size
    else:
        h_out = hidden_dim

    data = torch.empty(samples,
                       input_dim * sequence_dim
                       + h_out * d * num_layers
                       + hidden_dim * d * num_layers
                       + h_out * d * sequence_dim
                       + h_out * d * num_layers
                       + hidden_dim * d * num_layers)

    # TODO: populate data based on meaningful data
    for sample_idx in range(samples):
        data[sample_idx] = torch.rand(
            input_dim * sequence_dim
            + h_out * d * num_layers
            + hidden_dim * d * num_layers
            + h_out * d * sequence_dim
            + h_out * d * num_layers
            + hidden_dim * d * num_layers)

    # Construct lightning data module for the dataset
    data_module = VectorSeqDataModule(hparams_args, data)

    # create model
    model = plu.models.lstm.LSTMModel(**hparams)

    # create trainer
    trainer = pl.Trainer.from_argparse_args(hparams_args)

    # tune trainer
    trainer.tune(model, datamodule=data_module)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)

    # export model
    onnx_filepath = 'lstm_example.onnx'
    onnx_batch_size = 1
    input_sequence = torch.randn((onnx_batch_size, sequence_dim, input_dim))
    input_hidden_cell = (
        torch.randn((onnx_batch_size, d * num_layers, h_out)),
        torch.randn((onnx_batch_size, d * num_layers, hidden_dim))
    )
    model.to_onnx(onnx_filepath, (input_sequence, input_hidden_cell),
                  export_params=True)


if __name__ == '__main__':
    main()
