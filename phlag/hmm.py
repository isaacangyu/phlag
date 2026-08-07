import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from enum import Enum
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast

from jax import jit, lax
from jax.scipy.special import kl_div
from jax.scipy.optimize import minimize
from jax.nn import softmax, one_hot
from jax.lax import while_loop
from jaxtyping import Array, Float, Int, PyTree
from dynamax.utils.utils import pytree_sum
from dynamax.hidden_markov_model.inference import *
from dynamax.hidden_markov_model.models.abstractions import (
    HMM,
    HMMEmissions,
    HMMTransitions,
    HMMParameterSet,
    HMMPropertySet,
)
from dynamax.hidden_markov_model.models.initial import (
    StandardHMMInitialState,
    ParamsStandardHMMInitialState,
)
from dynamax.hidden_markov_model.models.transitions import ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties

PRNGKeyT = Array
Scalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]


def _normalize(u: Array, axis=0, eps=1e-15):
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


def _condition_on(probs, ll):
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(
    probs: Float[Array, " num states"], A: Float[Array, "num states num states"]
) -> Float[Array, "num states"]:
    return A.T @ probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_backward_filter(
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[int], Float[Array, "num_states num_states"]]] = None,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> Tuple[Scalar, Float[Array, "num_timesteps num_states"]]:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        """Backward filtering step."""
        log_normalizer, backward_pred_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t - 1)
        ll = log_likelihoods[t] + occupancy_bias
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        log_normalizer += log_norm
        next_backward_pred_probs = _predict(backward_filt_probs, A.T)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    (log_normalizer, _), backward_pred_probs = lax.scan(
        _step, (0.0, jnp.ones(num_states)), jnp.arange(num_timesteps), reverse=True
    )
    return log_normalizer, backward_pred_probs


@partial(jit, static_argnames=["transition_fn"])
def hmm_filter(
    initial_distribution: Float[Array, " num_states"],
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> HMMPosteriorFiltered:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t] + occupancy_bias
        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, A)
        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    carry = (0.0, initial_distribution)
    (log_normalizer, _), (filtered_probs, predicted_probs) = lax.scan(
        _step, carry, jnp.arange(num_timesteps)
    )

    post = HMMPosteriorFiltered(
        marginal_loglik=log_normalizer,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
    )
    return post


@partial(jit, static_argnames=["transition_fn"])
def hmm_two_filter_smoother(
    initial_distribution: Float[Array, " num_states"],
    transition_matrix: Optional[
        Union[
            Float[Array, "num_states num_states"],
            Float[Array, "num_timesteps_minus_1 num_states num_states"],
        ]
    ],
    log_likelihoods: Float[Array, "num_timesteps num_states"],
    transition_fn: Optional[Callable[[IntScalar], Float[Array, "num_states num_states"]]] = None,
    compute_trans_probs: bool = True,
    occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
) -> HMMPosterior:
    post = hmm_filter(
        initial_distribution, transition_matrix, log_likelihoods, transition_fn, occupancy_bias
    )
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    _, backward_pred_probs = hmm_backward_filter(
        transition_matrix, log_likelihoods, transition_fn, occupancy_bias
    )

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    posterior = HMMPosterior(
        marginal_loglik=ll,
        filtered_probs=filtered_probs,
        predicted_probs=predicted_probs,
        smoothed_probs=smoothed_probs,
        initial_probs=smoothed_probs[0],
    )

    # Compute the transition probabilities if specified
    if compute_trans_probs:
        trans_probs = compute_transition_probs(transition_matrix, posterior, transition_fn)
        posterior = posterior._replace(trans_probs=trans_probs)

    return posterior


def hellinger2_distance(p: Float[Array, "num_classes"], q: Float[Array, "num_classes"]) -> Float:
    return jnp.sum((jnp.sqrt(p) - jnp.sqrt(q)) ** 2) * 0.5


def divergence_e(e_0: Float[Array, "num_classes"], e_1: Float[Array, "num_classes"]) -> Float:
    return hellinger2_distance(e_1, e_0)


class ParamsBetaHMMEmissions(NamedTuple):
    concentration1: Union[Float[Array, "num_states emission_dim"], ParameterProperties]
    concentration0: Union[Float[Array, "num_states emission_dim"], ParameterProperties]


class PhlagBetaHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: int,
        emission_dim: int,
        parameterization: Tuple[str, ...],
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.parameterization = parameterization

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsBetaHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        c1 = jnp.clip(params.concentration1[state], a_min=1e-5)
        c0 = jnp.clip(params.concentration0[state], a_min=1e-5)
        return tfd.Independent(
            tfd.Beta(concentration1=c1, concentration0=c0),
            reinterpreted_batch_ndims=1
        )

    def log_prior(self, params: ParamsBetaHMMEmissions) -> Scalar:
        return 0.0

    def initialize(
        self,
        key: PRNGKeyT = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim 2"]] = None,
    ) -> Tuple[ParamsBetaHMMEmissions, ParamsBetaHMMEmissions]:
        if emission_probs is not None:
            means = jnp.clip(emission_probs[..., 0], a_min=1e-5, a_max=1.0 - 1e-5)
            variances = jnp.clip(emission_probs[..., 1], a_min=1e-5)
            v = means * (1.0 - means) / variances - 1.0
            v = jnp.clip(v, a_min=1e-2)
            c1 = means * v
            c0 = (1.0 - means) * v
        else:
            c1 = 2.0 * jnp.ones((self.num_states, self.emission_dim))
            c0 = 2.0 * jnp.ones((self.num_states, self.emission_dim))

        params = ParamsBetaHMMEmissions(concentration1=c1, concentration0=c0)
        props = ParamsBetaHMMEmissions(
            concentration1=ParameterProperties(constrainer=tfb.Softplus()),
            concentration0=ParameterProperties(constrainer=tfb.Softplus())
        )
        return params, props

    def collect_suff_stats(
        self, params: ParamsBetaHMMEmissions, posterior: HMMPosterior, emissions: Array, inputs=None
    ) -> dict:
        y = jnp.clip(emissions, a_min=1e-7, a_max=1.0 - 1e-7)
        sum_weights = jnp.sum(posterior.smoothed_probs, axis=0)
        sum_y = jnp.einsum("tk,td->kd", posterior.smoothed_probs, y)
        sum_y_sq = jnp.einsum("tk,td->kd", posterior.smoothed_probs, y ** 2)
        return dict(sum_weights=sum_weights, sum_y=sum_y, sum_y_sq=sum_y_sq)

    def initialize_m_step_state(
        self, params: ParamsBetaHMMEmissions, props: ParamsBetaHMMEmissions
    ) -> Any:
        return None

    def update_m_step_state(
        self, params: ParamsBetaHMMEmissions, props: ParamsBetaHMMEmissions
    ) -> Any:
        return None

    def state_divergence(self, params: ParamsBetaHMMEmissions) -> Float:
        c1 = jnp.clip(params.concentration1, a_min=1e-5)
        c0 = jnp.clip(params.concentration0, a_min=1e-5)
        means = c1 / (c1 + c0)
        total_divergence = 0.0
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                total_divergence += jnp.sqrt(jnp.sum((means[i] - means[j]) ** 2))
        return total_divergence

    def m_step(
        self,
        params: ParamsBetaHMMEmissions,
        props: ParamsBetaHMMEmissions,
        batch_stats: dict,
        m_step_state: Any,
    ) -> Tuple[ParamsBetaHMMEmissions, Any]:
        stats = pytree_sum(batch_stats, axis=0)
        sum_weights = jnp.clip(stats["sum_weights"][:, None], a_min=1e-6)
        sum_y = stats["sum_y"]
        sum_y_sq = stats["sum_y_sq"]

        means = jnp.clip(sum_y / sum_weights, a_min=1e-4, a_max=1.0 - 1e-4)
        variances = jnp.clip((sum_y_sq / sum_weights) - means ** 2, a_min=1e-6)

        max_var = 0.99 * means * (1.0 - means)
        variances = jnp.clip(variances, a_max=max_var)

        v = means * (1.0 - means) / variances - 1.0
        v = jnp.clip(v, a_min=1e-2, a_max=1000.0)

        c1 = jnp.clip(means * v, a_min=1e-2, a_max=1000.0)
        c0 = jnp.clip((1.0 - means) * v, a_min=1e-2, a_max=1000.0)

        params = params._replace(concentration1=c1, concentration0=c0)
        return params, m_step_state


class ParamsGMMHMMEmissions(NamedTuple):
    mixture_weights: Union[Float[Array, "num_states emission_dim num_mixtures"], ParameterProperties]
    means: Union[Float[Array, "num_states emission_dim num_mixtures"], ParameterProperties]
    stds: Union[Float[Array, "num_states emission_dim num_mixtures"], ParameterProperties]


class PhlagGMMHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: int,
        emission_dim: int,
        num_mixtures: int,
        parameterization: Tuple[str, ...],
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.num_mixtures = num_mixtures
        self.parameterization = parameterization

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsGMMHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        weights = jnp.clip(params.mixture_weights[state], a_min=1e-6)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        means = params.means[state]
        stds = jnp.clip(params.stds[state], a_min=1e-5)
        
        mix = tfd.Categorical(probs=weights)
        comp = tfd.Normal(loc=means, scale=stds)
        
        return tfd.Independent(
            tfd.MixtureSameFamily(mixture_distribution=mix, components_distribution=comp),
            reinterpreted_batch_ndims=1
        )

    def log_prior(self, params: ParamsGMMHMMEmissions) -> Scalar:
        return 0.0

    def initialize(
        self,
        key: PRNGKeyT = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim 2"]] = None,
        initial_gmm_params: Optional[Tuple[Array, Array, Array]] = None,
    ) -> Tuple[ParamsGMMHMMEmissions, ParamsGMMHMMEmissions]:
        S = self.num_states
        D = self.emission_dim
        M = self.num_mixtures
        
        if initial_gmm_params is not None:
            mixture_weights, means, stds = initial_gmm_params
            mixture_weights = jnp.array(mixture_weights, dtype=jnp.float32)
            means = jnp.array(means, dtype=jnp.float32)
            stds = jnp.array(stds, dtype=jnp.float32)
        else:
            if emission_probs is not None:
                state_means = emission_probs[..., 0]
                state_vars = emission_probs[..., 1]
                state_stds = jnp.sqrt(jnp.clip(state_vars, a_min=1e-5))
            else:
                state_means = jnp.zeros((S, D))
                state_stds = jnp.ones((S, D))
                
            if M == 1:
                offsets = jnp.array([0.0])
            else:
                offsets = jnp.linspace(-1.0, 1.0, M)
                
            means = state_means[:, :, None] + offsets[None, None, :] * state_stds[:, :, None] * 0.5
            stds = jnp.stack([state_stds for _ in range(M)], axis=-1)
            mixture_weights = jnp.ones((S, D, M)) / M
            
        params = ParamsGMMHMMEmissions(mixture_weights=mixture_weights, means=means, stds=stds)
        props = ParamsGMMHMMEmissions(
            mixture_weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            means=ParameterProperties(),
            stds=ParameterProperties(constrainer=tfb.Softplus())
        )
        return params, props

    def collect_suff_stats(
        self, params: ParamsGMMHMMEmissions, posterior: HMMPosterior, emissions: Array, inputs=None
    ) -> dict:
        T, D = emissions.shape
        M = self.num_mixtures
        
        sum_weights_list = []
        sum_y_list = []
        sum_y_sq_list = []
        
        for s in range(self.num_states):
            means = params.means[s]
            stds = jnp.clip(params.stds[s], a_min=1e-5)
            log_weights = jnp.log(jnp.clip(params.mixture_weights[s], a_min=1e-12))
            
            log_lik_comp = tfd.Normal(loc=means, scale=stds).log_prob(emissions[:, :, None])
            joint_log_lik = log_lik_comp + log_weights[None, :, :]
            
            comp_resp = jnp.exp(joint_log_lik - jax.scipy.special.logsumexp(joint_log_lik, axis=-1, keepdims=True))
            w_state = posterior.smoothed_probs[:, s]
            joint_weights = comp_resp * w_state[:, None, None]
            
            sum_weights_list.append(jnp.sum(joint_weights, axis=0))
            sum_y_list.append(jnp.sum(joint_weights * emissions[:, :, None], axis=0))
            sum_y_sq_list.append(jnp.sum(joint_weights * (emissions[:, :, None] ** 2), axis=0))
            
        return dict(
            sum_weights=jnp.stack(sum_weights_list, axis=0),
            sum_y=jnp.stack(sum_y_list, axis=0),
            sum_y_sq=jnp.stack(sum_y_sq_list, axis=0)
        )

    def initialize_m_step_state(
        self, params: ParamsGMMHMMEmissions, props: ParamsGMMHMMEmissions
    ) -> Any:
        return None

    def update_m_step_state(
        self, params: ParamsGMMHMMEmissions, props: ParamsGMMHMMEmissions
    ) -> Any:
        return None

    def state_divergence(self, params: ParamsGMMHMMEmissions) -> Float:
        overall_means = jnp.sum(params.mixture_weights * params.means, axis=-1)
        total_divergence = 0.0
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                total_divergence += jnp.sqrt(jnp.sum((overall_means[i] - overall_means[j]) ** 2))
        return total_divergence

    def m_step(
        self,
        params: ParamsGMMHMMEmissions,
        props: ParamsGMMHMMEmissions,
        batch_stats: dict,
        m_step_state: Any,
    ) -> Tuple[ParamsGMMHMMEmissions, Any]:
        stats = pytree_sum(batch_stats, axis=0)
        sum_weights = stats["sum_weights"]
        sum_y = stats["sum_y"]
        sum_y_sq = stats["sum_y_sq"]
        
        total_weights = jnp.clip(sum_weights.sum(axis=-1, keepdims=True), a_min=1e-12)
        mixture_weights = sum_weights / total_weights
        
        denom = jnp.clip(sum_weights, a_min=1e-8)
        means = sum_y / denom
        
        variances = jnp.clip((sum_y_sq / denom) - means ** 2, a_min=1e-6)
        stds = jnp.sqrt(variances)
        
        if hasattr(self, "mixture_masks"):
            mixture_weights = mixture_weights * self.mixture_masks
            mixture_weights = mixture_weights / jnp.clip(mixture_weights.sum(axis=-1, keepdims=True), a_min=1e-12)
            
        params = params._replace(
            mixture_weights=mixture_weights,
            means=means,
            stds=stds
        )
        return params, m_step_state


class PhlagHMMTransitions(HMMTransitions):
    def __init__(
        self,
        num_states: int,
        concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
    ):
        self.num_states = num_states
        self.concentration = concentration * jnp.ones((num_states, num_states))

    def distribution(
        self, params: ParamsStandardHMMTransitions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        return tfd.Categorical(probs=params.transition_matrix[state])

    def initialize(
        self,
        key: Optional[PRNGKeyT] = None,
        method: str = "prior",
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
    ) -> Tuple[ParamsStandardHMMTransitions, ParamsStandardHMMTransitions]:
        if transition_matrix is None:
            if method.lower() == "prior":
                if key is None:
                    raise ValueError("A key required if transition matrix not provided")
                tm_sample = tfd.Dirichlet(self.concentration).sample(seed=key)
                transition_matrix = cast(Float[Array, "num_states num_states"], tm_sample)
            else:
                raise Exception("Invalid initialization method: {}".format(method))
        else:
            assert transition_matrix.shape == (self.num_states, self.num_states)
        params = ParamsStandardHMMTransitions(transition_matrix=transition_matrix)
        props = ParamsStandardHMMTransitions(
            transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props

    def log_prior(self, params: ParamsStandardHMMTransitions) -> Scalar:
        return 0.0

    def _compute_transition_matrices(
        self, params: ParamsStandardHMMTransitions, inputs=None
    ) -> Float[Array, "num_states num_states"]:
        return params.transition_matrix

    def collect_suff_stats(
        self, params: ParamsStandardHMMTransitions, posterior: HMMPosterior, inputs=None
    ):
        return posterior.trans_probs

    def initialize_m_step_state(
        self, params: ParamsStandardHMMTransitions, props: ParamsStandardHMMTransitions
    ) -> Any:
        return None

    def m_step(
        self,
        params: ParamsStandardHMMTransitions,
        props: ParamsStandardHMMTransitions,
        batch_stats: Float[Array, "batch num_states num_states"],
        m_step_state: Any,
    ) -> Tuple[ParamsStandardHMMTransitions, Any]:
        if props.transition_matrix.trainable:
            if self.num_states == 1:
                transition_matrix = jnp.array([[1.0]])
            else:
                expected_trans_counts = batch_stats.sum(axis=0)
                transition_matrix = expected_trans_counts / (expected_trans_counts.sum(axis=-1, keepdims=True) + 1e-12)
            params = params._replace(transition_matrix=transition_matrix)
        return params, m_step_state


class ParamsGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "num_states emission_dim"], ParameterProperties]
    covariances: Union[Float[Array, "num_states emission_dim emission_dim"], ParameterProperties]


class PhlagHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: int,
        emission_dim: int,
        penalty_lambda: float,
        parameterization: Tuple[str, ...],
        concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.penalty_lambda = penalty_lambda
        self.parameterization = parameterization
        self.concentration = concentration

    def set_emission_prior_concentration(
        self, concentration: Union[Scalar, Float[Array, "num_classes"]]
    ):
        self.concentration = concentration
    
    def _ensure_concentration_shape(self, num_classes: int):
        """Ensure concentration has correct shape for the given num_classes."""
        if jnp.asarray(self.concentration).size == 1:
            self.concentration = self.concentration * jnp.ones(num_classes)
        elif jnp.asarray(self.concentration).size != num_classes:
            raise ValueError(
                f"Concentration size {jnp.asarray(self.concentration).size} does not match num_classes {num_classes}"
            )

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsGaussianHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        mean = params.means[state]
        # Full state covariance (captures cross-topology correlation), with a small
        # diagonal jitter for numerical stability/positive-definiteness -- the same
        # role the old diagonal clip played, plus a floor beneath the M-step's own
        # eps=1e-5 regularizer for states the M-step hasn't touched yet (e.g. the
        # very first E-step, before any M-step has run).
        cov = params.covariances[state] + jnp.eye(self.emission_dim) * 1e-6

        return tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)

    def log_prior(self, params: ParamsGaussianHMMEmissions) -> Scalar:
        # Return 0 for now (flat prior on continuous emissions)
        return 0.0

    def initialize(
        self,
        key: Optional[PRNGKeyT] = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]] = None,
    ) -> Tuple[ParamsGaussianHMMEmissions, ParamsGaussianHMMEmissions]:
        if emission_probs is None:
            raise ValueError("emission_probs must be provided")

        if emission_probs.ndim == 3 and emission_probs.shape[-1] == 2:
            num_states, emission_dim, _ = emission_probs.shape
            means = emission_probs[..., 0]
            variances = emission_probs[..., 1]
            covariances = jnp.stack(
                [jnp.diag(variances[state]) for state in range(num_states)], axis=0
            )
            params = ParamsGaussianHMMEmissions(means=means, covariances=covariances)
        elif emission_probs.ndim == 2:
            means = emission_probs
            num_states, emission_dim = means.shape
            covariances = jnp.stack([jnp.eye(emission_dim) for _ in range(num_states)], axis=0)
            params = ParamsGaussianHMMEmissions(means=means, covariances=covariances)
        else:
            raise ValueError(
                "Unsupported emission_probs shape. Expected [num_states, emission_dim, 2] "
                "for mean/variance initialization or [num_states, emission_dim] for means only."
            )

        props = ParamsGaussianHMMEmissions(
            means=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            covariances=ParameterProperties(constrainer=tfb.SoftmaxCentered())
        )
        return params, props
    
    def collect_suff_stats(self, params: ParamsGaussianHMMEmissions, posterior: Any, emissions: Array, inputs=None) -> dict:
        # posterior.smoothed_probs has shape (T, num_states)
        # emissions has shape (T, emission_dim)
        sum_weights = jnp.sum(posterior.smoothed_probs, axis=0)
        sum_x = jnp.einsum("tk,td->kd", posterior.smoothed_probs, emissions)
        sum_xxT = jnp.einsum("tk,ti,tj->kij", posterior.smoothed_probs, emissions, emissions)
        return dict(sum_weights=sum_weights, sum_x=sum_x, sum_xxT=sum_xxT)

    def initialize_m_step_state(
        self, params: ParamsGaussianHMMEmissions, props: ParamsGaussianHMMEmissions
    ) -> Any:
        return None

    def update_m_step_state(
        self, params: ParamsGaussianHMMEmissions, props: ParamsGaussianHMMEmissions
    ) -> Any:
        return None

    def m_step(
        self,
        params: ParamsGaussianHMMEmissions,
        props: ParamsGaussianHMMEmissions,
        batch_stats: dict,
        m_step_state: Any,
    ) -> Tuple[ParamsGaussianHMMEmissions, Any]:
        if props.means.trainable or props.covariances.trainable:
            stats = pytree_sum(batch_stats, axis=0)
            emission_stats = pytree_sum(batch_stats, axis=0)
            sum_weights = stats["sum_weights"][:, None] 
            sum_x = stats["sum_x"]                     
            sum_xxT = stats["sum_xxT"]                 

            # Analytical updates for continuous targets
            means = sum_x / (sum_weights + 1e-12)
            mu_muT = jnp.einsum("ki,kj->kij", means, means)
            covariances = (sum_xxT / (sum_weights[..., None] + 1e-12)) - mu_muT
        
            # Add regularizer matrix to guarantee positive-definiteness 
            eps = 1e-5
            covariances += jnp.stack([jnp.eye(self.emission_dim) * eps for _ in range(self.num_states)])
            params = params._replace(means=means, covariances=covariances)
        return params, m_step_state

    def state_divergence(self, params: ParamsGaussianHMMEmissions) -> Float:
        means = params.means
        # Compute mean divergence between states (Euclidean distance)
        total_divergence = 0
        for i in range(means.shape[0]):
            for j in range(i + 1, means.shape[0]):
                # L2 distance between means
                 total_divergence += jnp.sqrt(jnp.sum((means[i] - means[j]) ** 2))
        return total_divergence


class ParamsPhlagHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: Any


class PhlagHMM(HMM):
    def __init__(
        self,
        num_states: int = 2,
        emission_dim: int = 1,
        emission_lambda: Scalar = 1,
        emission_parameterization: Tuple[str, ...] = None,
        emission_concentration: Union[Scalar, Float[Array, "num_classes"]] = 1.1,
        initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
        occupancy_bias: Union[Scalar, Float[Array, "num_states"]] = 0.0,
        model_design: str = "gaussian",
        **kwargs,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_lambda = emission_lambda
        if emission_parameterization is not None:
            self.emission_parameterization = emission_parameterization
        else:
            self.emission_parameterization = ("free",) * self.num_states

        self.emission_concentration = emission_concentration
        self.initial_probs_concentration = initial_probs_concentration
        self.transition_concentration = transition_concentration

        self.initial_probs_m_step_state = None
        self.emissions_m_step_state = None
        self.transitions_m_step_state = None
        self.occupancy_bias = occupancy_bias
        # Cache for fit_em's compiled em_step closure, keyed on the identity of
        # the (props, emissions, inputs) it was built for -- see fit_em.
        self._jit_em_step = None
        self._jit_em_step_key = None

        self.initial_component = StandardHMMInitialState(
            num_states=self.num_states, initial_probs_concentration=self.initial_probs_concentration
        )
        self.transition_component = PhlagHMMTransitions(
            num_states=self.num_states, concentration=self.transition_concentration
        )
        if model_design == "beta":
            self.emission_component = PhlagBetaHMMEmissions(
                self.num_states,
                self.emission_dim,
                parameterization=self.emission_parameterization,
            )
        elif model_design == "gmm":
            num_mixtures = kwargs.get("num_mixtures", 2)
            self.emission_component = PhlagGMMHMMEmissions(
                self.num_states,
                self.emission_dim,
                num_mixtures=num_mixtures,
                parameterization=self.emission_parameterization,
            )
        else:
            self.emission_component = PhlagHMMEmissions(
                self.num_states,
                self.emission_dim,
                penalty_lambda=self.emission_lambda,
                concentration=self.emission_concentration,
                parameterization=self.emission_parameterization,
            )
        super().__init__(
            num_states=self.num_states,
            initial_component=self.initial_component,
            transition_component=self.transition_component,
            emission_component=self.emission_component,
        )

    def initialize(
        self,
        key: PRNGKeyT = jr.PRNGKey(0),
        method: str = "prior",
        emission_probs: Optional[Float[Array, "num_states emission_dim num_classes"]] = None,
        initial_probs: Optional[Float[Array, "num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
        initial_gmm_params: Optional[Tuple[Array, Array, Array]] = None,
    ) -> Tuple[ParamsPhlagHMM, ParamsPhlagHMM]:
        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        params["transitions"], props["transitions"] = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix
        )
        if isinstance(self.emission_component, PhlagGMMHMMEmissions):
            params["emissions"], props["emissions"] = self.emission_component.initialize(
                key3, method=method, emission_probs=emission_probs, initial_gmm_params=initial_gmm_params
            )
        else:
            params["emissions"], props["emissions"] = self.emission_component.initialize(
                key3, method=method, emission_probs=emission_probs
            )
        return ParamsPhlagHMM(**params), ParamsPhlagHMM(**props)

    def initialize_m_step_state(
        self,
        params: HMMParameterSet,
        props: HMMPropertySet,
        initial_probs_m_step_state=None,
        transitions_m_step_state=None,
        emissions_m_step_state=None,
    ) -> Tuple[Any, Any, Any]:
        if initial_probs_m_step_state is not None:
            self.initial_probs_m_step_state = initial_probs_m_step_state
        if transitions_m_step_state is not None:
            self.transitions_m_step_state = transitions_m_step_state
        if emissions_m_step_state is not None:
            self.emissions_m_step_state = emissions_m_step_state

        if self.initial_probs_m_step_state is None:
            self.initial_probs_m_step_state = self.initial_component.initialize_m_step_state(
                params.initial, props.initial
            )
        if self.transitions_m_step_state is None:
            self.transitions_m_step_state = self.transition_component.initialize_m_step_state(
                params.transitions, props.transitions
            )
        if self.emissions_m_step_state is None:
            self.emissions_m_step_state = self.emission_component.initialize_m_step_state(
                params.emissions, props.emissions
            )
        return (
            self.initial_probs_m_step_state,
            self.transitions_m_step_state,
            self.emissions_m_step_state,
        )

    def log_prior(self, params: HMMParameterSet) -> Scalar:
        return 0.0

    def state_emission_divergence(self, params: HMMParameterSet) -> Float[Array, "emission_dim"]:
        return self.emission_component.state_divergence(params.emissions)

    def m_step(
        self, params: HMMParameterSet, props: HMMPropertySet, batch_stats: PyTree, m_step_state: Any
    ) -> Tuple[HMMParameterSet, Any]:
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        initial_probs_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state

        initial_params, initial_probs_m_step_state = self.initial_component.m_step(
            params.initial, props.initial, batch_initial_stats, initial_probs_m_step_state
        )
        transition_params, transitions_m_step_state = self.transition_component.m_step(
            params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state
        )
        emission_params, emissions_m_step_state = self.emission_component.m_step(
            params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state
        )
        params = params._replace(
            initial=initial_params, transitions=transition_params, emissions=emission_params
        )
        m_step_state = (
            initial_probs_m_step_state,
            transitions_m_step_state,
            emissions_m_step_state,
        )
        return params, m_step_state

    def e_step(
        self,
        params: HMMParameterSet,
        emissions: Array,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[PyTree, Scalar]:
        args = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args, occupancy_bias=self.occupancy_bias)

        initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
        transition_stats = self.transition_component.collect_suff_stats(
            params.transitions, posterior, inputs
        )
        emission_stats = self.emission_component.collect_suff_stats(
            params.emissions, posterior, emissions, inputs
        )
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def fit_em(
        self,
        params,
        props,
        emissions,
        inputs=None,
        num_iters=50,
        verbose=True
    ):
        from dynamax.utils.utils import ensure_array_has_batch_dim

        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Phlag.run() calls fit_em ~10 times per training run, always with the
        # same (props, emissions/self.Y, inputs) objects. A fresh
        # @jax.jit-decorated closure built here every call would force a full
        # XLA retrace/recompile each time (jit's cache is keyed on the identity
        # of the jax.jit(...) call, not on function/value equality) -- so cache
        # the closure itself, keyed on the identity of what it closes over, and
        # rebuild it only if fit_em is ever called with different objects.
        #
        # This keeps props/batch_emissions/batch_inputs as closed-over
        # constants exactly like the original per-call closure did (rather
        # than passing them as traced jit arguments): that distinction matters
        # because XLA's constant-folding differs between "baked-in constant"
        # and "runtime argument" compilation, which shifts floating-point
        # rounding and compounds over ~275 iterative EM steps into a visibly
        # different converged result -- confirmed empirically, so this is not
        # a cosmetic choice.
        cache_key = (id(props), id(emissions), id(inputs))
        if self._jit_em_step is None or self._jit_em_step_key != cache_key:
            @jax.jit
            def em_step(params, m_step_state):
                batch_stats, lls = jax.vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
                lp = self.log_prior(params) + lls.sum()
                params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
                return params, m_step_state, lp

            self._jit_em_step = em_step
            self._jit_em_step_key = cache_key

        em_step = self._jit_em_step

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, props)

        if verbose:
            # Print initial transition matrix
            tm_init = params.transitions.transition_matrix
            tm_init_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in tm_init.tolist())
            print(f"Initial Transition matrix: {tm_init_str}")

        for step in range(num_iters):
            params, m_step_state, marginal_loglik = em_step(params, m_step_state)
            log_probs.append(marginal_loglik)

            if verbose:
                # Print transition probabilities at each iteration
                tm = params.transitions.transition_matrix
                tm_str = ", ".join(f"[{', '.join(f'{x:.6f}' for x in row)}]" for row in tm.tolist())
                print(f"EM iteration {step + 1}/{num_iters} - Transition matrix: {tm_str}")

        return params, jnp.array(log_probs)
