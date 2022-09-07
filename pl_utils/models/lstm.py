"""LSTM Model."""

import argparse
import ast

import pytorch_lightning as pl

import torch


# pylint: disable=too-many-ancestors
class LSTMModel(pl.LightningModule):
    """LSTM Network."""

    def __init__(self, **kwargs):
        """Initialize fully connected layers."""
        super().__init__()

        self.save_hyperparameters()

        self.lstm = torch.nn.LSTM(self.hparams.input_size,
                                  self.hparams.hidden_size,
                                  self.hparams.num_layers,
                                  self.hparams.bias,
                                  self.hparams.batch_first,
                                  self.hparams.dropout,
                                  self.hparams.bidirectional,
                                  self.hparams.proj_size)

    # pylint: disable=arguments-differ
    def forward(self, input_sequence, input_hidden_cell):
        """Compute prediction."""
        input_hidden, input_cell = input_hidden_cell

        # TODO: Can permuting be done at dataloader level to speed forward call
        # How to ensure we are getting the right batch dimensions stuff?

        # Restructure batch data to fit lstm model
        # input_sequence: we have [batch_dim, sequence_dim, input_dim], but
        # if self.hparams.batch_first
        # we need [batch_dim, sequence_dim, input_dim]
        # so do nothing
        # else
        # we need [sequence_dim, batch_dim, input_dim]
        if not self.hparams.batch_first:
            input_sequence = input_sequence.permute(1, 0, 2)

        # hidden/cell: we get [batch_dim, num_layer * d, hidden_dim], but
        # for lstm input we need [num_layer * d, batch_dim, hidden_dim]
        # batch_first=True only applies to input_sequence not hidden/cell
        input_hidden = input_hidden.permute(1, 0, 2).contiguous()
        input_cell = input_cell.permute(1, 0, 2).contiguous()

        output_sequence, (output_hidden, output_cell) = self.lstm(
            input_sequence, (input_hidden, input_cell))

        return output_sequence, (output_hidden, output_cell)

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        """Compute training loss."""
        loss = self.compute_loss(batch)
        return {'loss': loss}

    def configure_optimizers(self):
        """Create optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)

        return optimizer

    # pylint: disable=unused-argument
    def validation_step(self, batch, batch_idx):
        """Compute validation loss."""
        loss = self.compute_loss(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """Record validation loss."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    # pylint: disable=unused-argument
    def test_step(self, batch, batch_idx):
        """Compute testing loss."""
        loss = self.compute_loss(batch)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        """Record average test loss."""
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss)

    def compute_loss(self, batch):
        """Compute loss of batch."""
        input_sequence, (input_hidden, input_cell), \
            target_sequence, (target_hidden, target_cell) = batch

        output_sequence, (output_hidden, output_cell) = self(input_sequence,
                                                             (input_hidden,
                                                              input_cell))

        # input_sequence/hidden/cell is permuted inside of the forward call
        target_hidden = target_hidden.permute(1, 0, 2).contiguous()
        target_cell = target_cell.permute(1, 0, 2).contiguous()

        sequence_loss = torch.nn.functional.mse_loss(output_sequence,
                                                     target_sequence)
        hidden_loss = torch.nn.functional.mse_loss(output_hidden,
                                                   target_hidden)
        cell_loss = torch.nn.functional.mse_loss(output_cell, target_cell)

        loss = sequence_loss + hidden_loss + cell_loss

        return loss

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """Parse model specific hyperparameters."""
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, required=True)
        parser.add_argument('--hidden_size', type=int, required=True)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--bias', type=ast.literal_eval, default=True)
        parser.add_argument('--batch_first', type=ast.literal_eval,
                            default=False)
        parser.add_argument('--dropout', type=float, default=0)
        parser.add_argument('--bidirectional', type=ast.literal_eval,
                            default=False)
        parser.add_argument('--proj_size', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=1e-3)

        return parser
