"""LSTM Model."""

import argparse

import pytorch_lightning as lt

import torch


# pylint: disable=too-many-ancestors
class LSTMModel(lt.core.lightning.LightningModule):
    """LSTM Network."""

    def __init__(self, **kwargs):
        """Initialize fully connected layers."""
        super().__init__()

        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(self.hparams.input_size,
                                  self.hparams.hidden_size,
                                  self.hparams.num_layers,
                                  batch_first=self.hparams.batch_first)

    # pylint: disable=arguments-differ
    def forward(self, input_sequence, initial_hidden):
        """Compute prediction."""
        initial_hidden_state, initial_cell_state = initial_hidden

        # from the batch we get <batch_dim, num_layer_dim, state_dim>, but
        # for lstm input we need <num_layer_dim, batch_dim, state_dim>
        # batch_first=True only applies to input not hidden/cell states
        initial_hidden_state = \
            initial_hidden_state.permute(1, 0, 2).contiguous()
        initial_cell_state = initial_cell_state.permute(1, 0, 2).contiguous()

        output, (_, _) = self.lstm(input_sequence, (initial_hidden_state,
                                                    initial_cell_state))
        return output

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        """Compute training loss."""
        input_sequence, (initial_hidden_state, initial_cell_state), target \
            = batch
        output = self(input_sequence, (initial_hidden_state,
                                       initial_cell_state))

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
        input_sequence, (initial_hidden_state, initial_cell_state), target \
            = batch
        output = self(input_sequence, (initial_hidden_state,
                                       initial_cell_state))

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
        input_sequence, (initial_hidden_state, initial_cell_state), target \
            = batch
        output = self(input_sequence, (initial_hidden_state,
                                       initial_cell_state))

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
        parser.add_argument('--input_size', type=int, required=True)
        parser.add_argument('--hidden_size', type=int, required=True)
        parser.add_argument('--num_layers', type=int, required=True)
        parser.add_argument('--batch_first', type=bool, default=True)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--momentum_param', type=int, default=0)

        return parser
