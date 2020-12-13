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

    def __init__(self, hparams, data):
        """Initialze variables."""
        super().__init__()
        self.hparams = hparams

        self.data = data
        # For GPs the batch_size must be the entire dataset
        self.batch_size = len(self.data)

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

        # Shuffle data
        random_indices = torch.randperm(self.data.shape[0])
        self.data = self.data[random_indices]

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
        return torch.utils.data.DataLoader(
            dataset=train_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs):
        """Create val dataloader."""
        val_split = VectorDataset(self.val_input_data, self.val_output_data)
        return torch.utils.data.DataLoader(
            dataset=val_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs):
        """Create test dataloader."""
        test_split = VectorDataset(self.test_input_data, self.test_output_data)
        return torch.utils.data.DataLoader(
            dataset=test_split,
            num_workers=self.hparams.data_num_workers,
            batch_size=self.batch_size)

    @staticmethod
    def _split_input_output_data(data):
        input_dim = 1
        input_data = data[:, 0:input_dim]
        output_data = data[:, input_dim:None]
        return input_data, output_data


# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
def main():
    """Initialize model and trainer to fit."""
    parser = argparse.ArgumentParser()

    # Add program specific args from model
    parser.add_argument('--data_num_workers', type=int, default=1)

    # Add trainer specific args from model
    parser = lt.Trainer.add_argparse_args(parser)

    # Add model specific args from model
    parser = plu.models.gp.BIMOEGPModel.add_model_specific_args(parser)

    program_args_list = ['--data_num_workers', '4']

    training_args_list = ['--auto_lr_find', 'False',
                          '--benchmark', 'True',
                          '--fast_dev_run', '0',
                          '--gpus', '-1',
                          '--logger', 'False',  # pytorch-lightning/#4496
                          '--max_epochs', '50',
                          '--terminate_on_nan', 'True',
                          '--weights_summary', 'full']

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
    data_module = RandomDataModule(hparams_args, sinusoidal_data)

    # GP data preprocessing since it needs train_input_data, train_output_data
    # during initialization which also need to be the same ones used in
    # training
    data_module.setup()

    # create model
    model = plu.models.gp.BIMOEGPModel(data_module.train_input_data,
                                       data_module.train_output_data,
                                       **hparams)

    # create trainer
    trainer = lt.Trainer.from_argparse_args(hparams_args)

    # tune trainer
    trainer.tune(model, datamodule=data_module)

    # train on data
    trainer.fit(model, datamodule=data_module)

    # test on data
    trainer.test(model, datamodule=data_module)

    # create some input data to be predicted on
    samples = 1000
    predict_input_data = torch.linspace(0, x_domain, samples).unsqueeze(1)
    predict_input_data = predict_input_data.cuda()

    # load model
    # gotta save the train data somehow prob just need to add a custom
    # checkpoint callback
    # checkpoint_path = 'lightning_logs/version_40/checkpoints/epoch=999.ckpt'
    # model = plu.models.gp.BIMOEGPModel.load_from_checkpoint(checkpoint_path)

    # predict on mode
    model.eval()
    with torch.no_grad():
        predictions = model.likelihood(model(predict_input_data))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # move over data to the cpu for plotting
    predict_input_data = predict_input_data.cpu()
    mean = mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()

    # plots
    # pylint: disable=import-outside-toplevel
    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=torch.sin(predict_input_data[:, 0] * freq),
            mode='lines',
            line={
                'color': 'blue',
                'dash': 'dot',
            },
            name='sin func',
            visible=False,
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data_module.train_input_data[:, 0],
            y=data_module.train_output_data[:, 0],
            mode='markers',
            marker={
                'size': 10,
                #'color': 'blue',
            },
            # name='sin data',
            name='data samples',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=mean[:, 0],
            mode='lines',
            # line={
            #     'color': 'blue',
            # },
            # name='sin mean',
            name='posterior mean',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=upper[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin upper',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=lower[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin lower',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=torch.cat([predict_input_data[:, 0],
                         torch.flip(predict_input_data[:, 0], [0])]),
            y=torch.cat([upper[:, 0], torch.flip(lower[:, 0], [0])]),
            mode='lines',
            line={
                'color': 'rgba(255, 255, 255, 0)',
            },
            fillcolor='black',
            opacity=0.2,
            fill='toself',
            # name='sin confidence',
            name='confidence region',
            showlegend=True
        )
    )

    """
    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=torch.cos(predict_input_data[:, 0] * freq),
            mode='lines',
            line={
                'color': 'red',
                'dash': 'dot',
            },
            name='cos func',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data_module.train_input_data[:, 0],
            y=data_module.train_output_data[:, 1],
            mode='markers',
            marker={
                'color': 'red',
            },
            name='cos data',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=mean[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            name='cos mean',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=upper[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            opacity=0,
            name='cos upper',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=lower[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            opacity=0,
            name='cos lower',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=torch.cat([predict_input_data[:, 0],
                         torch.flip(predict_input_data[:, 0], [0])]),
            y=torch.cat([upper[:, 1], torch.flip(lower[:, 1], [0])]),
            mode='lines',
            line={
                'color': 'rgba(255, 255, 255, 0)',
            },
            fillcolor='red',
            opacity=0.2,
            fill='toself',
            name='cos confidence',
            showlegend=True
        )
    )
    """

    fig.update_layout(
        title='Gaussian Process Regression',
        xaxis_title='Input',
        yaxis_title='Observation',
        font={
            'size': 18,
        },
    )

    fig.show()


if __name__ == '__main__':
    main()
