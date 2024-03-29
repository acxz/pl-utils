"""Gaussian Process Model."""

import argparse

import gpytorch

import pytorch_lightning as pl

import torch


class BIMOEGP(gpytorch.models.ExactGP):
    """batch independent multioutput exact gp model."""

    def __init__(self, train_input_data, train_output_data, likelihood):
        """Initialize gp model with mean and covar."""
        super().__init__(train_input_data, train_output_data, likelihood)

        output_dim = train_output_data.size(dim=1)
        output_dim_torch = torch.Size([output_dim])

        self.mean_module = \
            gpytorch.means.ConstantMean(batch_shape=output_dim_torch)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=output_dim_torch),
            batch_shape=output_dim_torch)

        # Receiving error when using jit:
        # RuntimeError: mean shape torch.Size([9, 2]) is incompatible with covariance shape torch.Size([144, 144])
        # Something wrong with prediction stage

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Compute prediction."""
        mean = self.mean_module(input_)
        covar = self.covar_module(input_)

        return \
            gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean, covar))


class MeanVarModelWrapper(torch.nn.Module):
    """Wrapper class to output prediction model."""

    def __init__(self, gp):
        """Initialize gp."""
        super().__init__()
        self.gp = gp

    def forward(self, input_):
        """Compute prediction."""
        output_dist = self.gp(input_)
        return output_dist.mean, output_dist.variance


# pylint: disable=too-many-ancestors
class BIMOEGPModel(pl.LightningModule):
    """batch independent multioutput exact gp model."""

    def __init__(self, train_input_data, train_output_data, **kwargs):
        """Initialize gp model with mean and covar."""
        super().__init__()

        self.save_hyperparameters()

        output_dim = self.hparams.train_output_data.shape[1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=output_dim)

        self.bimoegp = BIMOEGP(self.hparams.train_input_data,
                               self.hparams.train_output_data,
                               self.likelihood)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.bimoegp)

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Compute prediction."""
        return self.bimoegp(input_)

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        """Compute training loss."""
        input_, target = batch
        output = self(input_)

        loss = -self.mll(output, target)

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
        input_, target = batch
        output = self(input_)

        loss = -self.mll(output, target)

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

        loss = -self.mll(output, target)

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
        parser.add_argument('--learning_rate', type=float, default=1e-3)

        return parser
