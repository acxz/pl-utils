"""Train a model from a training dataset."""

import argparse
import math

import pl_utils as plu

import pytorch_lightning as pl

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
class VectorDataModule(pl.core.datamodule.LightningDataModule):
    """Data module to load train/val/test dataloaders."""

    def __init__(self, hparams, data):
        """Initialze variables."""
        super().__init__()
        self.save_hyperparameters(hparams)

        self.data = data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Create and assign splits."""
        train_pct = 0.8
        test_pct = 0.1

        samples = len(self.data)

        train_samples = int(train_pct * samples)
        test_samples = int(test_pct * samples)
        val_samples = samples - train_samples - test_samples

        input_data, output_data = self._split_input_output_data(self.data)
        vector_dataset = VectorDataset(input_data, output_data)
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                vector_dataset, [train_samples, val_samples, test_samples])

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
    def _split_input_output_data(data):
        input_dim = 1
        input_data = data[:, 0:input_dim]
        output_data = data[:, input_dim:None]
        return input_data, output_data


# pylint: disable=too-many-locals
def main():
    """Initialize model and trainer to fit a fc net."""
    parser = argparse.ArgumentParser()

    # Add program specific args from model
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_num_workers', type=int, default=1)

    # Add trainer specific args from model
    parser = pl.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.fc.FCModel.add_model_specific_args(parser)

    program_args_list = ['--batch_size', '8000',
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
                          '--max_epochs', '1000',
                          '--enable_model_summary', 'True']

    model_args_list = ['--layer_dims', '1 100 100 3',
                       '--learning_rate', '0.002']

    args_list = program_args_list + training_args_list + model_args_list

    hparams_args = parser.parse_args(args_list)
    hparams = vars(hparams_args)

    # Create data to fit a fully connected network
    samples = 100000
    input_dim = 1
    output_dim = 3
    data = torch.empty(samples, input_dim + output_dim)

    time = torch.linspace(1, 5.7, samples)
    for sample_idx in range(samples):
        data[sample_idx] = torch.Tensor([time[sample_idx],
                                         torch.cos(2 * torch.pi *
                                                   time[sample_idx]),
                                         torch.sin(2 * torch.pi *
                                                   time[sample_idx]),
                                         time[sample_idx]])

    # Construct lightning data module for the dataset
    data_module = VectorDataModule(hparams_args, data)

    # create model
    model = plu.models.fc.FCModel(**hparams)

    # create trainer
    trainer = pl.Trainer.from_argparse_args(hparams_args)

    # tune trainer
    trainer.tune(model, datamodule=data_module)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)

    # export model
    onnx_filepath = 'fc_example.onnx'
    onnx_batch_size = 1
    input_sample = torch.randn((onnx_batch_size, input_dim))
    model.to_onnx(onnx_filepath, input_sample, export_params=True)


if __name__ == '__main__':
    main()
