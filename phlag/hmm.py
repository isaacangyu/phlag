import math

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from enum import Enum
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union, cast

from jax import jit, lax
from jax.flatten_util import ravel_pytree
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
) -> Tuple[Scalar, Float[Array, "num_timesteps num_states"]]:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        """Backward filtering step."""
        log_normalizer, backward_pred_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t - 1)
        ll = log_likelihoods[t]
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
) -> HMMPosteriorFiltered:
    num_timesteps, num_states = log_likelihoods.shape

    def _step(carry, t):
        log_normalizer, predicted_probs = carry
        A = get_trans_mat(transition_matrix, transition_fn, t)
        ll = log_likelihoods[t]
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
) -> HMMPosterior:
    post = hmm_filter(
        initial_distribution, transition_matrix, log_likelihoods, transition_fn
    )
    ll = post.marginal_loglik
    filtered_probs, predicted_probs = post.filtered_probs, post.predicted_probs

    _, backward_pred_probs = hmm_backward_filter(
        transition_matrix, log_likelihoods, transition_fn
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

def gaussian_bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2):
    d = mu1.shape[0]
    eps = 1e-6
    Sigma_avg = 0.5 * (Sigma1 + Sigma2) + eps * jnp.eye(d)
    diff = mu1 - mu2
    _, logdet1 = jnp.linalg.slogdet(Sigma1 + eps * jnp.eye(d))
    _, logdet2 = jnp.linalg.slogdet(Sigma2 + eps * jnp.eye(d))
    _, logdet_avg = jnp.linalg.slogdet(Sigma_avg)
    log_coef = 0.25 * logdet1 + 0.25 * logdet2 - 0.5 * logdet_avg
    quad = diff @ jnp.linalg.solve(Sigma_avg, diff)
    return jnp.exp(log_coef - 0.125 * quad)


def gaussian_hellinger2(mu1, Sigma1, mu2, Sigma2):
    return 1.0 - gaussian_bhattacharyya_coefficient(mu1, Sigma1, mu2, Sigma2)


def gaussian_hellinger_distance(mu1, Sigma1, mu2, Sigma2):
    """Hellinger distance, bounded [0, 1]: 0 = identical distributions, 1 = fully separated."""
    return jnp.sqrt(jnp.clip(gaussian_hellinger2(mu1, Sigma1, mu2, Sigma2), a_min=0.0))


class ParamsGMMHMMEmissions(NamedTuple):
    mixture_weights: Union[Float[Array, "num_states num_mixtures"], ParameterProperties]
    means: Union[Float[Array, "num_states num_mixtures emission_dim"], ParameterProperties]
    covariances: Union[Float[Array, "num_states num_mixtures emission_dim emission_dim"], ParameterProperties]


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
        weights = weights / weights.sum()
        means = params.means[state]  # [num_mixtures, emission_dim]
        covariances = params.covariances[state] + jnp.eye(self.emission_dim) * 1e-6

        mix = tfd.Categorical(probs=weights)
        comp = tfd.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covariances)

        return tfd.MixtureSameFamily(mixture_distribution=mix, components_distribution=comp)

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
            mixture_weights, means, covariances = initial_gmm_params
            mixture_weights = jnp.array(mixture_weights, dtype=jnp.float32)
            means = jnp.array(means, dtype=jnp.float32)
            covariances = jnp.array(covariances, dtype=jnp.float32)
        else:
            if emission_probs is not None:
                state_means = emission_probs[..., 0]
                state_vars = emission_probs[..., 1]
            else:
                state_means = jnp.zeros((S, D))
                state_vars = jnp.ones((S, D))
            state_stds = jnp.sqrt(jnp.clip(state_vars, a_min=1e-5))

            state_cov = jnp.stack(
                [jnp.diag(state_vars[s]) for s in range(S)], axis=0
            )

            if M == 1:
                offsets = jnp.array([0.0])
            else:
                offsets = jnp.linspace(-1.0, 1.0, M)

            # means: [S, M, D], each mixture component offset along the diagonal spread
            means = state_means[:, None, :] + offsets[None, :, None] * state_stds[:, None, :] * 0.5
            covariances = jnp.stack([state_cov for _ in range(M)], axis=1)  # [S, M, D, D]
            mixture_weights = jnp.ones((S, M)) / M

        params = ParamsGMMHMMEmissions(mixture_weights=mixture_weights, means=means, covariances=covariances)
        props = ParamsGMMHMMEmissions(
            mixture_weights=ParameterProperties(constrainer=tfb.SoftmaxCentered()),
            means=ParameterProperties(),
            covariances=ParameterProperties()
        )
        return params, props

    def collect_suff_stats(
        self, params: ParamsGMMHMMEmissions, posterior: HMMPosterior, emissions: Array, inputs=None
    ) -> dict:
        T, D = emissions.shape

        sum_weights_list = []
        sum_x_list = []
        sum_xxT_list = []

        for s in range(self.num_states):
            means = params.means[s]  # [M, D]
            covariances = params.covariances[s] + jnp.eye(D) * 1e-6  # [M, D, D]
            log_weights = jnp.log(jnp.clip(params.mixture_weights[s], a_min=1e-12))  # [M]

            comp = tfd.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covariances)
            log_lik_comp = comp.log_prob(emissions[:, None, :])  # [T, M]
            joint_log_lik = log_lik_comp + log_weights[None, :]

            comp_resp = jnp.exp(joint_log_lik - jax.scipy.special.logsumexp(joint_log_lik, axis=-1, keepdims=True))
            w_state = posterior.smoothed_probs[:, s]
            joint_weights = comp_resp * w_state[:, None]  # [T, M]

            sum_weights_list.append(jnp.sum(joint_weights, axis=0))  # [M]
            sum_x_list.append(jnp.einsum("tm,td->md", joint_weights, emissions))  # [M, D]
            sum_xxT_list.append(jnp.einsum("tm,ti,tj->mij", joint_weights, emissions, emissions))  # [M, D, D]

        return dict(
            sum_weights=jnp.stack(sum_weights_list, axis=0),
            sum_x=jnp.stack(sum_x_list, axis=0),
            sum_xxT=jnp.stack(sum_xxT_list, axis=0)
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
        overall_means = jnp.sum(params.mixture_weights[..., None] * params.means, axis=1)
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
        sum_weights = stats["sum_weights"]  # [S, M]
        sum_x = stats["sum_x"]  # [S, M, D]
        sum_xxT = stats["sum_xxT"]  # [S, M, D, D]

        total_weights = jnp.clip(sum_weights.sum(axis=-1, keepdims=True), a_min=1e-12)
        mixture_weights = sum_weights / total_weights

        denom = jnp.clip(sum_weights, a_min=1e-8)
        means = sum_x / denom[..., None]

        mean_outer = jnp.einsum("smi,smj->smij", means, means)
        covariances = sum_xxT / denom[..., None, None] - mean_outer
        covariances = covariances + jnp.eye(self.emission_dim) * 1e-6

        if hasattr(self, "mixture_masks"):
            mixture_weights = mixture_weights * self.mixture_masks
            mixture_weights = mixture_weights / jnp.clip(mixture_weights.sum(axis=-1, keepdims=True), a_min=1e-12)

        params = params._replace(
            mixture_weights=mixture_weights,
            means=means,
            covariances=covariances
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
            transition_matrix=ParameterProperties()
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


class EmissionParam(Enum):
    FREE = "free"
    REPULSION = "repulsion"


class PhlagHMMEmissions(HMMEmissions):
    def __init__(
        self,
        num_states: int,
        emission_dim: int,
        penalty_lambda: float,
        parameterization: Tuple[str, ...],
        lm_damping: float = 1.0,
        repulsion_optimizer: str = "lm",
        penalty_lambda_anneal: bool = False,
        anneal_boost: float = 2.0,
        n_iters: int = 10,
        increment_steps: int = 5,
    ):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.penalty_lambda = penalty_lambda
        self.parameterization = parameterization
        self.lm_damping = lm_damping
        self.repulsion_optimizer = repulsion_optimizer
        self.penalty_lambda_anneal = penalty_lambda_anneal
        self.anneal_boost = anneal_boost
        # t (in exp(-t/anneal_tau), see m_step) is counted in actual inner EM
        # steps completed, not outer-loop iterations -- phlag.py's outer loop
        # runs (i+1)*increment_steps inner steps at outer index i, so total
        # steps over the whole run is the arithmetic-sequence sum below, not
        # n_iters itself. tau is a third of that total.
        total_inner_steps = increment_steps * n_iters * (n_iters + 1) // 2
        self.anneal_tau = max(1, total_inner_steps // 6)

        # "Budget mode": un-normalized, the schedule below (1 + boost*exp(-t/tau))
        # is always >= 1, so penalty_lambda * that is always >= penalty_lambda --
        # --lam would only ever be a floor the schedule decays toward, never
        # its average. Instead, precompute the schedule's own time-weighted
        # mean (weighted by how many inner EM steps run at each t, since t is
        # held fixed for a whole outer iteration's num_inner steps) and divide
        # it out, so effective_lambda's average over the whole run equals
        # exactly penalty_lambda -- --lam becomes a budget spent unevenly
        # across iterations (elevated early, reduced later) rather than a floor.
        # Pure Python/math here (not jnp): n_iters/increment_steps/anneal_boost
        # are all known at construction time, no need to trace this.
        weighted_exp_sum = 0.0
        cumulative = 0
        for i in range(n_iters):
            num_inner = (i + 1) * increment_steps
            weighted_exp_sum += math.exp(-cumulative / self.anneal_tau) * num_inner
            cumulative += num_inner
        mean_exp = weighted_exp_sum / total_inner_steps if total_inner_steps > 0 else 0.0
        mean_schedule = 1.0 + anneal_boost * mean_exp
        self.anneal_budget_scale = 1.0 / mean_schedule if mean_schedule > 0 else 1.0

    @property
    def emission_shape(self) -> Tuple[int]:
        return (self.emission_dim,)

    def distribution(
        self, params: ParamsGaussianHMMEmissions, state: IntScalar, inputs=None
    ) -> tfd.Distribution:
        mean = params.means[state]
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

        if emission_probs.ndim != 3:
            raise ValueError(
                "Unsupported emission_probs shape. Expected [num_states, emission_dim, 2] "
                "for mean/variance initialization, or [num_states, emission_dim, emission_dim + 1] "
                "for mean/full-covariance initialization."
            )

        num_states, emission_dim, num_classes = emission_probs.shape
        means = emission_probs[..., 0]

        if num_classes == 2:
            variances = emission_probs[..., 1]
            covariances = jnp.stack(
                [jnp.diag(variances[state]) for state in range(num_states)], axis=0
            )
        elif num_classes == emission_dim + 1:
            covariances = emission_probs[..., 1:]
        else:
            raise ValueError(
                "Unsupported emission_probs shape. Expected [num_states, emission_dim, 2] "
                "for mean/variance initialization, or [num_states, emission_dim, emission_dim + 1] "
                "for mean/full-covariance initialization."
            )

        params = ParamsGaussianHMMEmissions(means=means, covariances=covariances)

        props = ParamsGaussianHMMEmissions(
            means=ParameterProperties(),
            covariances=ParameterProperties()
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
        # (annealing clock t, cumulative LM step-norm clip activations,
        # cumulative LM attempts taken) -- t is only read when
        # penalty_lambda_anneal is on, but the clip counters are always
        # tracked so fit_em can surface gradient-clip diagnostics regardless.
        return (jnp.array(0.0), jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

    def update_m_step_state(
        self, params: ParamsGaussianHMMEmissions, props: ParamsGaussianHMMEmissions
    ) -> Any:
        return None

    def map_estimate_repulsion(self, mu_target, Sigma_target, n, xbar, Sxx, penalty_lambda=None):
        penalty_lambda = self.penalty_lambda if penalty_lambda is None else penalty_lambda
        d = xbar.shape[0]
        eps = 1e-6
        # precomputed before optimization
        S = Sxx - n * jnp.outer(xbar, xbar)

        def neg_objective(params):
            # abstract params
            mu, L = params # joint update
            Sigma = L @ L.T + eps * jnp.eye(d)
            Sigma_inv = jnp.linalg.inv(Sigma)
            diff = xbar - mu
            weighted_scatter_about_mu = S + n * jnp.outer(diff, diff)

            ll = -0.5 * (
                n * jnp.linalg.slogdet(Sigma)[1]
                + jnp.trace(Sigma_inv @ weighted_scatter_about_mu)
            )
            # ll scales with n (every term above is n-weighted), but bc doesn't
            # scale with occupancy at all -- at a fixed penalty_lambda the
            # penalty is swamped by ll once n is more than a handful, so
            # repulsion silently degenerates to FREE for any well-populated
            # state. Scale the effective penalty by n so it stays proportionate
            # to ll regardless of occupancy.
            effective_lambda = penalty_lambda * n
            bc = gaussian_bhattacharyya_coefficient(mu, Sigma, mu_target, Sigma_target)
            return -(ll - effective_lambda * bc)  # repulsion: reward small BC, i.e. large Hellinger distance

        mu0 = xbar
        L0 = jnp.linalg.cholesky(Sigma_target + eps * jnp.eye(d))
        flat0, unravel = ravel_pytree((mu0, L0))
        dim = flat0.shape[0]

        def flat_obj(flat):
            return neg_objective(unravel(flat))

        grad_flat = jax.grad(flat_obj)

        sigma_min = 1e-2

        def clamp_L(L):
            # Eigendecompose Sigma (symmetric by construction) and floor its
            # eigenvalues at sigma_min before re-Choleskying, instead of only
            # relying on the +eps*I floor -- directly prevents the
            # near-singular-Sigma -> exploding-gradient cascade found earlier.
            Sigma = L @ L.T + eps * jnp.eye(d)
            eigvals, eigvecs = jnp.linalg.eigh(Sigma)
            eigvals = jnp.clip(eigvals, a_min=sigma_min)
            Sigma_clamped = (eigvecs * eigvals) @ eigvecs.T
            return jnp.linalg.cholesky(Sigma_clamped)

        if self.repulsion_optimizer == "gd":
            # Plain fixed-step gradient descent -- a genuinely separate route
            # rather than a limiting case of the LM step below: damping=0
            # there is an undamped Newton step (uses the full Hessian), and
            # damping=inf is numerically degenerate in the (H + damping*I)
            # solve, so neither end of that dial reaches this. Still clamps
            # Sigma's eigenvalues each step like the LM path, since that floor
            # guards against the same near-singular-Sigma blowup regardless of
            # which optimizer is stepping.
            gd_lr = 0.05

            def gd_step(_, flat):
                mu_i, L_i = unravel(flat)
                flat = ravel_pytree((mu_i, clamp_L(L_i)))[0]
                g = grad_flat(flat)
                return flat - gd_lr * g

            # lax.fori_loop instead of a Python loop so the 100 steps don't
            # unroll into the jit trace of the outer EM step.
            flat_final = lax.fori_loop(0, 100, gd_step, flat0)
            # Plain GD has no step-norm clip -- the concept doesn't apply to
            # this path, so it contributes no clip activations or attempts.
            clip_count = jnp.array(0, dtype=jnp.int32)
            clip_attempts = jnp.array(0, dtype=jnp.int32)
        else:
            hess_flat = jax.hessian(flat_obj)

            # Levenberg-Marquardt (damped Newton): solves (H + damping*I)
            # delta = -g each step, retrying with damping *= 10 (falls back
            # toward a small gradient-descent-like step) until the step
            # doesn't make neg_objective worse/nan, and relaxing damping
            # (*= 0.5, floored) after an accepted step so it can speed back up
            # toward full Newton steps when the landscape allows it. Tuned/
            # verified against the same 10X/down/N109/40-60 leaf that reliably
            # diverged/NaN'd under the earlier plain-gradient-descent scheme
            # -- converges cleanly for both single- and both-state REPULSION.
            max_step_norm = 10.0
            max_lm_tries = 20
            damping0 = self.lm_damping

            def step(_, carry):
                flat, damping, clip_count, clip_attempts = carry
                g = grad_flat(flat)
                H = hess_flat(flat)
                obj0 = flat_obj(flat)

                def lm_try(damping):
                    A = H + damping * jnp.eye(dim)
                    delta = jnp.linalg.solve(A, -g)
                    # Clip the step norm directly -- bounds step *magnitude*, not
                    # just objective value, the same gap plain backtracking had.
                    delta_norm = jnp.linalg.norm(delta)
                    clip_scale = jnp.minimum(1.0, max_step_norm / (delta_norm + 1e-12))
                    clipped = (clip_scale < 1.0).astype(jnp.int32)
                    delta = delta * clip_scale
                    flat_new = flat + delta
                    obj_new = flat_obj(flat_new)
                    return flat_new, obj_new, clipped

                def cond_fn(state):
                    damping, n_tries, _, obj_new, _ = state
                    worse = jnp.logical_or(jnp.isnan(obj_new), obj_new > obj0)
                    return jnp.logical_and(worse, n_tries < max_lm_tries)

                def body_fn(state):
                    damping, n_tries, _, _, clip_acc = state
                    damping = damping * 10.0
                    flat_new, obj_new, clipped = lm_try(damping)
                    return (damping, n_tries + 1, flat_new, obj_new, clip_acc + clipped)

                flat_new0, obj_new0, clipped0 = lm_try(damping)
                damping_f, n_tries_f, flat_f, _, clip_acc_f = lax.while_loop(
                    cond_fn, body_fn,
                    (damping, jnp.array(0, dtype=jnp.int32), flat_new0, obj_new0, clipped0),
                )
                damping_next = jnp.maximum(damping_f * 0.5, 1e-7)

                mu_new, L_new = unravel(flat_f)
                L_new = clamp_L(L_new)
                flat_clamped, _ = ravel_pytree((mu_new, L_new))
                # n_tries_f counts only the while_loop's retries -- +1 accounts
                # for the initial lm_try that seeded its carry before any
                # retry ran, so every step contributes at least one attempt.
                return (flat_clamped, damping_next, clip_count + clip_acc_f, clip_attempts + n_tries_f + 1)

            # lax.fori_loop instead of a Python loop so the 100 steps don't
            # unroll into the jit trace of the outer EM step.
            flat_final, _, clip_count, clip_attempts = lax.fori_loop(
                0, 100, step,
                (flat0, damping0, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)),
            ) # why fixed at 100?

        mu, L = unravel(flat_final)

        Sigma = L @ L.T + eps * jnp.eye(d)
        return mu, Sigma, clip_count, clip_attempts

    def m_step(self, params, props, batch_stats, m_step_state):
        t, clip_count, clip_attempts = m_step_state
        if props.means.trainable or props.covariances.trainable:
            stats = pytree_sum(batch_stats, axis=0)
            sum_weights, sum_x, sum_xxT = stats["sum_weights"], stats["sum_x"], stats["sum_xxT"]

            # null params repel from previous iteration's alt params, and vice versa
            old_means, old_covariances = params.means, params.covariances
            new_means, new_covariances = old_means, old_covariances

            if self.penalty_lambda_anneal:
                # t holds the cumulative inner-EM-step count completed before
                # this fit_em call (held fixed for this whole call's inner
                # loop -- see PhlagHMM.fit_em/phlag.py's outer loop, which passes
                # that cumulative count in as outer_iter despite the name).
                # Budget mode (see anneal_budget_scale in __init__): elevated
                # early, reduced later, so the run-long average lands on
                # penalty_lambda exactly -- see map_estimate_repulsion for what
                # this multiplies.
                penalty_lambda_t = self.penalty_lambda * self.anneal_budget_scale * (
                    1.0 + self.anneal_boost * jnp.exp(-t / self.anneal_tau)
                )
            else:
                penalty_lambda_t = self.penalty_lambda

            for s in range(self.num_states):
                n_s = sum_weights[s]
                xbar_s = sum_x[s] / (n_s + 1e-12)
                Sxx_s = sum_xxT[s]
                mode = EmissionParam(self.parameterization[s])

                if mode is EmissionParam.FREE:
                    mu_s = xbar_s
                    Sigma_s = Sxx_s / (n_s + 1e-12) - jnp.outer(xbar_s, xbar_s) + 1e-5 * jnp.eye(self.emission_dim)
                elif mode is EmissionParam.REPULSION:
                    other = 1 - s # assumes 2 states
                    mu_s, Sigma_s, clip_count_s, clip_attempts_s = self.map_estimate_repulsion(
                        old_means[other], old_covariances[other], n_s, xbar_s, Sxx_s,
                        penalty_lambda=penalty_lambda_t,
                    )
                    clip_count = clip_count + clip_count_s
                    clip_attempts = clip_attempts + clip_attempts_s

                new_means = new_means.at[s].set(mu_s)
                new_covariances = new_covariances.at[s].set(Sigma_s)

            params = params._replace(means=new_means, covariances=new_covariances)
        m_step_state = (t, clip_count, clip_attempts)
        return params, m_step_state

    def state_divergence(self, params: ParamsGaussianHMMEmissions) -> Float:
        """Mean L2 divergence between states (Euclidean distance between means)."""
        means = params.means
        total_divergence = 0
        for i in range(means.shape[0]):
            for j in range(i + 1, means.shape[0]):
                total_divergence += jnp.sqrt(jnp.sum((means[i] - means[j]) ** 2))
        return total_divergence

    def em_divergence(self, params: ParamsGaussianHMMEmissions) -> Float:
        """
        Squared Hellinger distance between states' fitted Gaussians, averaged
        over every unordered state pair -- bounded to [0, 1], where 0 means
        the two states are statistically identical and 1 means fully
        separated (higher is better separation). Unlike state_divergence
        (means only, the report's "State divergence" line), this folds both
        the mean AND the full covariance difference into a single number.
        The same Bhattacharyya coefficient map_estimate_repulsion optimizes
        against (Hellinger^2 = 1 - BC) -- squared rather than plain Hellinger
        (which would take a further sqrt) since sqrt compresses the spread
        near 1, bunching highly-separated cases together with little
        resolution between them.
        """
        means = params.means
        covariances = params.covariances
        num_states = means.shape[0]
        total_distance = 0.0
        num_pairs = 0
        for i in range(num_states):
            for j in range(i + 1, num_states):
                dist = gaussian_hellinger2(
                    means[i], covariances[i], means[j], covariances[j]
                )
                total_distance += dist
                num_pairs += 1
        return total_distance / num_pairs


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
        initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_concentration: Union[Scalar, Float[Array, "num_states num_states"]] = 1.1,
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

        self.initial_probs_concentration = initial_probs_concentration
        self.transition_concentration = transition_concentration

        self.initial_probs_m_step_state = None
        self.emissions_m_step_state = None
        self.transitions_m_step_state = None
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
        if model_design == "gmm":
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
                parameterization=self.emission_parameterization,
                lm_damping=kwargs.get("lm_damping", 1.0),
                repulsion_optimizer=kwargs.get("repulsion_optimizer", "lm"),
                penalty_lambda_anneal=kwargs.get("penalty_lambda_anneal", False),
                n_iters=kwargs.get("n_iters", 10),
                increment_steps=kwargs.get("increment_steps", 5),
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

    def em_divergence(self, params: HMMParameterSet) -> Float:
        """Squared Hellinger distance between states -- gaussian-only, since
        it depends on emission_component.em_divergence (see PhlagHMMEmissions)."""
        return self.emission_component.em_divergence(params.emissions)

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
        posterior = hmm_two_filter_smoother(*args)

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
        verbose=True,
        outer_iter=None,
    ):
        from dynamax.utils.utils import ensure_array_has_batch_dim

        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)


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
        if outer_iter is not None:
            # Clip counters always start fresh for this fit_em call (matches
            # the non-annealed branch below) -- only the annealing clock t is
            # explicitly seeded from outer_iter.
            m_step_state = self.initialize_m_step_state(
                params, props,
                emissions_m_step_state=(
                    jnp.array(float(outer_iter)),
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(0, dtype=jnp.int32),
                ),
            )
        else:
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

        # Gradient clip diagnostics only exist for gaussian/REPULSION fits --
        # PhlagGMMHMMEmissions.m_step passes emissions_m_step_state through
        # untouched (always None), so there's nothing to report for gmm.
        _, _, emissions_m_step_state = m_step_state
        if isinstance(self.emission_component, PhlagHMMEmissions):
            _, clip_count, clip_attempts = emissions_m_step_state
            clip_count, clip_attempts = int(clip_count), int(clip_attempts)
        else:
            clip_count, clip_attempts = 0, 0

        return params, jnp.array(log_probs), clip_count, clip_attempts
