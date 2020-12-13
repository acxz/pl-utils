"""Fully Connected Model."""

import argparse
from collections import OrderedDict

import pytorch_lightning as lt

import torch


# pylint: disable=too-many-ancestors
class FCModel(lt.core.lightning.LightningModule):
    """FC Network."""

    def __init__(self, **kwargs):
        """Initialize fully connected layers."""
        super().__init__()

        self.save_hyperparameters()

        layer_dims_list_str = self.hparams.layer_dims.split()
        layer_dims = [int(layer_dim)
                      for layer_dim in layer_dims_list_str]
        fc_layers = OrderedDict()

        for layer_idx, _ in enumerate(layer_dims[0:-2]):
            fc_layers['linear' + str(layer_idx + 1)] = \
                torch.nn.Linear(layer_dims[layer_idx],
                                layer_dims[layer_idx + 1])
            fc_layers['selu' + str(layer_idx + 1)] = torch.nn.SELU()

        layer_idx = len(layer_dims) - 2
        fc_layers['linear' + str(layer_idx + 1)] = \
            torch.nn.Linear(layer_dims[layer_idx],
                            layer_dims[layer_idx + 1])

        self.model = torch.nn.Sequential(fc_layers)

        self.example_input_array = torch.randn(1, layer_dims[0])

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Compute prediction."""
        return self.model(input_)

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        """Compute training loss."""
        input_, target = batch
        output = self(input_)

        loss = torch.nn.functional.mse_loss(output, target)

        return {'loss': loss}

    def configure_optimizers(self):
        """Create optimizer."""
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum_param)

        return optimizer

    # pylint: disable=unused-argument
    def validation_step(self, batch, batch_idx):
        """Compute validation loss."""
        input_, target = batch
        output = self(input_)

        loss = torch.nn.functional.mse_loss(output, target)

        return {'val_loss': loss}

    # pylint: disable=no-self-use
    def validation_epoch_end(self, outputs):
        """Record validation loss."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    # pylint: disable=unused-argument
    def test_step(self, batch, batch_idx):
        """Compute testing loss."""
        input_, target = batch
        output = self(input_)

        loss = torch.nn.functional.mse_loss(output, target)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        """Record average test loss."""
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """Parse model specific hyperparameters."""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--layer_dims', type=str, required=True)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--momentum_param', type=int, default=0)

        return parser
