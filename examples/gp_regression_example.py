"""Train a model from a training dataset."""

import argparse
import math

import gpytorch

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

        # flag to ensure setup (i.e. random split) only happens once
        self.has_setup = False

        self.data = data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Create and assign splits."""
        if self.has_setup:
            return
        self.has_setup = True

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
            # batch size must be the training dataset for GPs
            batch_size=len(self.train_dataset))

    def val_dataloader(self, *args, **kwargs):
        """Create val dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self, *args, **kwargs):
        """Create test dataloader."""
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def _split_input_output_data(data):
        input_dim = 1
        input_data = data[:, 0:input_dim]
        output_data = data[:, input_dim:None]
        return input_data, output_data


# pylint: disable=too-many-locals
def main():
    """Initialize model and trainer to fit."""
    parser = argparse.ArgumentParser()

    # Add program specific args from model
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--data_num_workers', type=int, default=1)

    # Add trainer specific args from model
    parser = pl.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.gp.BIMOEGPModel.add_model_specific_args(parser)

    program_args_list = ['--eval_batch_size', '1',
                         '--data_num_workers', '4']

    training_args_list = ['--accelerator', 'auto',
                          '--accumulate_grad_batches', '1',
                          '--auto_lr_find', 'False',
                          '--benchmark', 'True',
                          '--enable_checkpointing', 'True',
                          '--detect_anomaly', 'True',
                          '--fast_dev_run', 'False',
                          '--enable_progress_bar', 'True',
                          '--max_epochs', '50',
                          '--enable_model_summary', 'True']

    model_args_list = ['--learning_rate', '0.832']

    args_list = program_args_list + training_args_list + model_args_list

    hparams_args = parser.parse_args(args_list)
    hparams = vars(hparams_args)

    # Create data to fit
    torch.manual_seed(3)
    samples = 100
    noise_scale = 0.05
    freq = 4 * math.pi
    x_domain = 1
    input_data = torch.linspace(0, x_domain, samples).unsqueeze(1)
    sin_output_data = torch.sin(input_data * freq) + \
        torch.randn(input_data.size()) * noise_scale
    cos_output_data = torch.cos(input_data * freq) + \
        torch.randn(input_data.size()) * noise_scale
    sinusoidal_data = torch.cat([input_data, sin_output_data, cos_output_data],
                                1)
    random_indices = torch.randperm(sinusoidal_data.shape[0])
    sinusoidal_data = sinusoidal_data[random_indices]
    sinusoidal_data = sinusoidal_data[0:10]

    # Construct lightning data module for the dataset
    data_module = VectorDataModule(hparams_args, sinusoidal_data)

    # setup data module manually since it needs train_input/output_data during
    # initialization, which also need to be the same ones used in training
    data_module.setup()
    train_input_data, train_output_data = data_module.train_dataset[:]

    # create model
    model = plu.models.gp.BIMOEGPModel(train_input_data,
                                       train_output_data,
                                       **hparams)

    # create trainer
    trainer = pl.Trainer.from_argparse_args(hparams_args)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)

    # export model
    pt_filepath = 'gp_regression_example.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'train_input_data': train_input_data,
                'train_output_data': train_output_data},
               pt_filepath)

    # export model in onnx
    # below does not work yet
    # onnx_filepath = 'gp_regression_example.onnx'
    # onnx_batch_size = 1
    # input_dim = 1
    # input_sample = torch.randn((onnx_batch_size, input_dim))

    # model.to_onnx(onnx_filepath, input_sample, export_params=True)

    # with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
    #        gpytorch.settings.trace_mode():
    #    model.eval()
    #    model(input_sample)  # Do precomputation

    #    traced_model = torch.jit.trace(
    #        plu.models.gp.MeanVarModelWrapper(model), input_sample)

    # traced_model = torch.jit.onnx(onnx_filepath, input_sample,
    #                          export_params=True)


if __name__ == '__main__':
    main()
