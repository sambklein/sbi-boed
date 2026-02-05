import matplotlib.pyplot as plt
import torch
from nflows import distributions, transforms
from nflows.flows import Flow
from sbi.neural_nets.estimators import NFlowsFlow
from nflows.utils import torchutils
import numpy as np
from sbi.utils.nn_utils import get_numel
from sbi_bax.models.conditional_linear import (
    ConditionalLuLinear,
    EmbeddingNet,
    MLPNetwork,
)
from sbi_bax.models.contex_only_transforms import (
    ContextOnlyPiecewiseRationalQuadraticTransform,
)
import torch.nn as nn


from typing import Union

from nflows.distributions.base import Distribution


class BoxUniform(Distribution):
    def __init__(
        self, low: Union[torch.Tensor, float], high: Union[torch.Tensor, float]
    ):
        """Multidimensionqal uniform distribution defined on a box.

        Args:
            low (Tensor or float): lower range (inclusive).
            high (Tensor or float): upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
        """
        super().__init__()

        if not torch.is_tensor(low):
            low = torch.tensor(low, dtype=torch.float32)
        if not torch.is_tensor(high):
            high = torch.tensor(high, dtype=torch.float32)

        if low.shape != high.shape:
            raise ValueError("low and high are not of the same size")

        if not (low < high).byte().all():
            raise ValueError("low has elements that are higher than high")

        self._shape = low.shape
        # self._low = low
        # self._high = high
        self.register_buffer("_low", low, persistent=False)
        self.register_buffer("_high", high, persistent=False)
        # self._log_z = -torch.sum(torch.log(high - low))
        self.register_buffer(
            "_log_z", -torch.sum(torch.log(high - low)), persistent=False
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self._log_z.expand(inputs.shape[0]).to(inputs.device)

    def _sample(self, num_samples, context):
        context_size = 1 if context is None else context.shape[0]
        low_expanded = self._low.expand(context_size * num_samples, *self._shape)
        high_expanded = self._high.expand(context_size * num_samples, *self._shape)
        samples = low_expanded + torch.rand(
            context_size * num_samples, *self._shape
        ).to(low_expanded.device) * (high_expanded - low_expanded)

        if context is None:
            return samples
        else:
            return torchutils.split_leading_dim(samples, [context_size, num_samples])


def spline_inn(
    inp_dim,
    nodes,
    num_blocks=2,
    nstack=3,
    tail_bound=None,
    tails=None,
    num_bins=10,
    context_features=None,
    input_transform=None,
    output_transform=None,
    identity_init=False,
):
    transform_list = []
    for i in range(nstack):
        # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
        # the final layer
        tpass = tails
        if tails is not None:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                inp_dim,
                nodes,
                num_blocks=num_blocks,
                tail_bound=tb,
                num_bins=num_bins,
                tails=tpass,
                context_features=context_features,
            )
        ]
        if identity_init:
            # Initialize to identity
            last_linear = transform_list[-1].autoregressive_net.final_layer
            last_linear.weight.data.fill_(0.0)
            last_linear.bias.data.fill_(
                np.log(np.exp(1 - transform_list[-1].min_derivative) - 1)
            )
        if (tails is None) and (tail_bound is not None) and (i == nstack - 1):
            transform_list += [
                transforms.standard.PointwiseAffineTransform(
                    -tail_bound, 2 * tail_bound
                )
            ]
        transform_list += [transforms.ReversePermutation(inp_dim)]

    # Drop the last permutation if nstack is odd, preserves "order" of variables
    if nstack % 2 == 1 and identity_init:
        transform_list = transform_list[:-1]

    if input_transform is not None:
        transform_list = [input_transform] + transform_list
    if output_transform is not None:
        transform_list = transform_list + [output_transform]

    return transforms.CompositeTransform(transform_list)


def check_density_model(
    x_vals,
    theta_true,
    flow,
    y_scaler,
    ctxt_scaler,
    noise_sim,
    figname,
    device,
    make_measurement,
):
    # Visualize the density function by looking at flow samples as a function of x for the true theta
    # Generate samples from the flow model
    n_samples = 10  # Number of sample per x value
    # Sample x_values from the x_vals
    x_samples = x_vals.repeat(n_samples, 1)  # Repeat x_vals for each sample
    # Scale the x values and theta values for the context
    ctxt = torch.cat((x_samples, theta_true.repeat(len(x_samples), 1)), dim=-1)
    # Normalize the context using the scaler
    ctxt_normalized = ctxt_scaler.transform(ctxt.cpu().numpy()).astype(np.float32)
    ctxt = torch.torch.Tensor(ctxt_normalized, dtype=torch.float32).to(device)
    # Sample from the flow model
    with torch.no_grad():
        flow_samples = flow.sample(1, context=ctxt)
    # Invert the normalization of the y values
    flow_samples = y_scaler.inverse_transform(
        flow_samples.cpu().numpy().reshape(-1, 1)
    ).astype(np.float32)
    # Plot the flow samples
    plt.figure(figsize=(10, 6))
    plt.scatter(x_samples.cpu().numpy(), flow_samples, alpha=0.1, label="Flow Samples")
    plt.scatter(
        x_samples.cpu().numpy(),
        make_measurement(x_samples, theta_true, noise_sim).cpu().numpy(),
        color="red",
        label="True Function",
    )
    plt.title("Flow Samples as a Function of x for True Theta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(figname)


class Nflow(Flow):
    """Do not condition on the embedded context in the base distribution."""

    def __init__(
        self, transform, distribution, embedding_net=None, base_context_op=None
    ):
        super().__init__(transform, distribution, embedding_net=embedding_net)
        if base_context_op is None:
            base_context_op = torch.nn.Identity()
        # Add option to operate independently on the context passed to the base distribution
        self.base_context = base_context_op

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(
            noise, context=self.base_context(context).detach()
        )
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Sometimes embedded context is fixed.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=context
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise


class SbiFlow(NFlowsFlow):
    def __init__(
        self,
        batch_obs,
        batch_theta,
        transform,
        distribution,
        embedding_net,
        base_context_op=None,
    ):
        super().__init__(
            Nflow(
                transform, distribution, embedding_net, base_context_op=base_context_op
            ),
            batch_theta[0].shape,
            batch_obs[0].shape,
        )


class LinearFlow(SbiFlow):
    def __init__(
        self,
        batch_obs: torch.torch.Tensor,
        batch_theta: torch.torch.Tensor,
        dataset_embeddor,
        n_layers: int,
        context_encoder: EmbeddingNet | None = None,
        identity_init=False,
        input_transform=None,
    ):
        """
        This flow will model data as a multimodal Gaussian. The main reason to implement this is because the entropy can
        be computed exactly for this model.
        dataset_embeddor: The dataset embedding network. This is defined globally with local contexts for each layer provided by context_encoder.
        """
        if context_encoder is None:
            context_encoder = MLPNetwork
        self.input_size = batch_theta[0].numel()
        # Each layer will separately encode the output of the dataset_embeddor
        context_dim = get_numel(batch_obs, embedding_net=dataset_embeddor)
        # Build list of transformations
        transform_list = [
            ConditionalLuLinear(
                self.input_size,
                context_dim,
                context_encoder,
                identity_init=identity_init,
            )
            for _ in range(n_layers)
        ]
        if input_transform is not None:
            # This is used to handle data normalization
            transform_list = [input_transform] + transform_list
        # This is the transformation that will map from data to the base distribution
        transform = transforms.CompositeTransform(transform_list)
        base_dist = distributions.StandardNormal([self.input_size])
        # Cast to float32 so can use MPS
        base_dist._log_z = base_dist._log_z.float()
        super(LinearFlow, self).__init__(
            batch_obs, batch_theta, transform, base_dist, dataset_embeddor
        )

    @property
    def device(self):
        return self.net._distribution._log_z.device

    def entropy_mc(self, context, n_samples):
        # No error in this model
        return self.entropy(context, n_samples=n_samples)

    def entropy(self, context, n_samples=None):
        # First we can get the exact entropy of the base distribution, a multivariate gaussian
        base_entropy = (
            0.5 * self.input_size * (1 + np.log(2 * np.pi))
        )  # log volume of the base distribution
        # Encode the full dataset
        dataset_embedding = self.net._embedding_net(context.to(self.device))
        # Get the entropy contribution of the transformations
        dummy_input = self.net._distribution.sample(context.shape[0])
        _, transform_entropy = self.net._transform.inverse(
            dummy_input, context=dataset_embedding
        )
        return base_entropy + transform_entropy


def pure_context_inn(
    input_size: int,
    context_dim: int,
    context_encoder=None,
    tail_bound=3.5,
    tails="linear",
    n_layers=5,
    input_transform=None,
    output_transform=None,
):
    if context_encoder is None:
        context_encoder = MLPNetwork
    # This is the transformation that will map from data to the base distribution
    transform_list = sum(
        [
            [
                ConditionalLuLinear(
                    input_size,
                    context_dim,
                    context_encoder,
                ),
                ContextOnlyPiecewiseRationalQuadraticTransform(
                    input_size,
                    context_dim,
                    context_encoder,
                    tails=tails,
                    tail_bound=tail_bound,
                ),
            ]
            for _ in range(n_layers)
        ],
        [],
    )
    if input_transform is not None:
        # This is used to handle normalization of the data
        transform_list = [input_transform] + transform_list
    if output_transform is not None:
        transform_list = transform_list + [output_transform]
    return transforms.CompositeTransform(transform_list)


class RqsPureContextFlow(SbiFlow):
    def __init__(
        self,
        batch_obs: torch.torch.Tensor,
        batch_theta: torch.torch.Tensor,
        dataset_embeddor,
        n_layers: int,
        context_encoder: EmbeddingNet | None = None,
        identity_init=False,
        tail_bound=3.5,
        input_transform=None,
    ):
        """
        This flow will model data as a multimodal Gaussian. The main reason to implement this is because the entropy can
        be computed exactly for this model.
        dataset_embeddor: The dataset embedding network. This is defined globally with local contexts for each layer provided by context_encoder.
        """
        self.input_size = batch_theta[0].numel()
        # Each layer will separately encode the output of the dataset_embeddor
        context_dim = get_numel(batch_obs, embedding_net=dataset_embeddor)
        # Get the transform list
        transform = pure_context_inn(
            self.input_size,
            context_dim,
            context_encoder=context_encoder,
            n_layers=n_layers,
            tail_bound=tail_bound,
            input_transform=input_transform,
        )
        base_dist = distributions.StandardNormal([self.input_size])
        # Cast to float32 so can use MPS
        base_dist._log_z = base_dist._log_z.float()
        super().__init__(batch_obs, batch_theta, transform, base_dist, dataset_embeddor)

    @property
    def device(self):
        return self.net._distribution._log_z.device


class GeneralFlow(SbiFlow):
    def __init__(
        self,
        batch_obs: torch.torch.Tensor,
        batch_theta: torch.torch.Tensor,
        dataset_embeddor,
        transform,
        uniform_base=False,
    ):
        """
        This flow will model data as a multimodal Gaussian. The main reason to implement this is because the entropy can
        be computed exactly for this model.
        dataset_embeddor: The dataset embedding network. This is defined globally with local contexts for each layer provided by context_encoder.
        """
        self.input_size = batch_theta[0].numel()
        # Each layer will separately encode the output of the dataset_embeddor
        context_dim = get_numel(batch_obs, embedding_net=dataset_embeddor)
        # This is the transformation that will map from data to the base distribution
        # This should handle data normalization directly
        transform = transform(self.input_size, context_dim)
        if uniform_base:
            base_dist = BoxUniform(
                low=-3.5 * torch.ones(self.input_size),
                high=3.5 * torch.ones(self.input_size),
            )
        else:
            base_dist = distributions.StandardNormal([self.input_size])
        # Cast to float32 so can use MPS
        base_dist._log_z = base_dist._log_z.float()
        super().__init__(batch_obs, batch_theta, transform, base_dist, dataset_embeddor)

    @property
    def device(self):
        return self.net._distribution._log_z.device


class StackedFlow(SbiFlow):
    def __init__(
        self,
        batch_obs: torch.torch.Tensor,
        batch_theta: torch.torch.Tensor,
        transform,
        base_flow,
        dataset_embeddor=None,
        base_context_op=None,
    ):
        # NOTE: this isn't needed because the log_prob method above uses no_grad
        # for param in base_flow.parameters():
        #     param.requires_grad = False
        super().__init__(
            batch_obs,
            batch_theta,
            transform,
            base_flow,
            dataset_embeddor,
            base_context_op=base_context_op,
        )
        self.base_flow = base_flow


class NuisanceFlow(nn.Module):
    def __init__(self, transform, distribution, dataset_embeddor, theta_embeddor):
        super().__init__()
        self.dataset_embeddor = dataset_embeddor
        self.flow = Nflow(transform, distribution)
        self.theta_embeddor = theta_embeddor

    def _prepare_context(self, theta, data):
        dataset_embedding = self.dataset_embeddor(data)
        theta_embedding = self.theta_embeddor(theta)
        return torch.cat((theta_embedding, dataset_embedding), dim=1)

    def log_prob(self, nuisance, theta, data):
        context = self._prepare_context(theta, data)
        return self.flow.log_prob(nuisance, context=context)

    def sample(self, theta, data, num_samples=1):
        context = self._prepare_context(theta, data)
        return self.flow.sample(context=context, num_samples=num_samples).squeeze()


class FixedOutout(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, context):
        n_samples = context.shape[0] if context is not None else 1
        return self.output.repeat_interleave(n_samples, 0)


class DeconditionedFlow(Nflow):
    """
    Once observation is fixed and model trained, encoding the observation into the flow context is fixed.
    """

    def __init__(self, flow, condition):
        super().__init__(
            flow._transform,
            flow._distribution,
            embedding_net=FixedOutout(flow._embedding_net(condition).detach()),
        )

    def sample(self, num_samples: tuple | int, context=None):
        # Override expected tuple input
        if isinstance(num_samples, tuple):
            num_samples = num_samples[0]
        return super().sample(num_samples, context=None)

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=context
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet
