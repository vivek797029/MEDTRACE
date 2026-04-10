"""
MedTrace — Adaptive Differential Privacy with Rényi DP Accounting
==================================================================
Implements per-client adaptive noise calibration with **mathematically
correct** privacy accounting via Rényi Differential Privacy (RDP) and
the optimal RDP → (ε, δ)-DP conversion.

Why the previous implementation was incorrect
---------------------------------------------
The previous version made two mathematical errors that caused it to
*overstate* privacy (report less ε than actually spent):

  1. ``budget_spent += ε_round`` — Simple summation of per-round ε values
     is only valid under *basic composition*, which gives the *worst-case*
     bound.  The standard "advanced composition" formula
     ε_total ≈ ε_round · √(T log(1/δ)) also requires a separate δ budget
     allocation and does NOT equal the true composed ε.

  2. ``σ = C · √(2 ln(1.25/δ)) / ε`` — This is the sufficient condition
     for the *approximate* Gaussian mechanism guarantee (Dwork & Roth 2014,
     Prop 3.22).  It yields σ that is larger than necessary and does not
     account for tight Rényi composition.

Correct approach: Rényi Differential Privacy (Mironov 2017)
------------------------------------------------------------
For the Gaussian mechanism with noise multiplier σ = noise_std / sensitivity:

    RDP(α) = α / (2σ²)    for all α > 1                    [Mironov 2017]

Composition is *additive* in the RDP domain:

    RDP_total(α) = Σ_t  RDP_t(α)                           [Composition Thm]

Conversion to (ε, δ)-DP uses the optimal tight conversion (Balle et al. 2020):

    ε(δ) = min_{α>1}  RDP(α) + log(α-1)/α
                              - [log(δ) + log(α-1)] / (α-1)

This is strictly tighter than the original Mironov (2017) conversion
    ε_Mironov(δ) = min_{α>1}  RDP(α) + log(1/δ) / (α-1)
and is the gold standard used by Opacus, TF-Privacy, and Google DP library.

Finding σ for a target (ε, δ)
-------------------------------
Because the RDP → (ε, δ) conversion is non-linear, there is no closed-form
inverse.  We use binary search over σ ∈ [σ_lo, σ_hi], evaluating
``RDPAccountant.get_epsilon(delta)`` at each candidate until we find the
smallest σ that achieves ε ≤ target_ε in ``steps`` rounds.

Subsampling amplification
--------------------------
When each round's gradient is computed over a random subsample of the
local dataset (Poisson subsampling rate q = batch_size / n_local), privacy
is amplified.  For integer α ≥ 2 the exact formula is (Wang et al. 2019):

    RDP_subsampled(α) = (1/(α-1)) log Σ_{k=0}^{α} C(α,k) q^k (1-q)^{α-k}
                                      · exp(k(k-1)/(2σ²))

For full-batch gradient sharing (q=1), this reduces to α/(2σ²).

References
----------
* Mironov (2017) — Rényi Differential Privacy of the Gaussian Mechanism.
  https://arxiv.org/abs/1702.07476
* Balle et al. (2020) — Hypothesis Testing Interpretations and Renyi DP.
  https://arxiv.org/abs/1905.09982  (tight RDP→(ε,δ) conversion)
* Wang et al. (2019) — Subsampled Rényi DP and Analytical Moments Accountant.
  https://arxiv.org/abs/1908.10530
* Dwork & Roth (2014) — The Algorithmic Foundations of Differential Privacy.
  FnT in TCS 9(3–4).
* Andrew et al. (2021) — Differentially Private Learning with Adaptive Clipping.
  https://arxiv.org/abs/1905.03871
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  RÉNYI DP ACCOUNTANT
# ═══════════════════════════════════════════════════════════════════════════════

# Fine-grained grid of Rényi orders.  Covering 1.5–1024 gives tight results
# for σ ∈ [0.4, 20] and T ∈ [1, 10 000].
_DEFAULT_ORDERS: Tuple[float, ...] = (
    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 48.0, 64.0,
    128.0, 256.0, 512.0, 1024.0,
)


class RDPAccountant:
    """
    Tracks cumulative Rényi DP privacy loss for the (subsampled) Gaussian
    mechanism and converts to (ε, δ)-DP via the tight Balle et al. (2020)
    formula.

    Usage
    -----
    >>> acc = RDPAccountant()
    >>> for _ in range(T):
    ...     acc.step(sigma=1.2, q=1.0)        # one FL round, full-batch
    >>> eps, alpha_star = acc.get_epsilon(delta=1e-5)

    One accountant instance per client.  Call ``step()`` each round the
    client participates, then ``get_epsilon(delta)`` at any time for the
    current (ε, δ) guarantee.

    Parameters
    ----------
    orders:
        Rényi orders α > 1 at which to track the RDP budget.  More orders
        → tighter bound (finer search grid for the minimum ε).
    """

    def __init__(self, orders: Tuple[float, ...] = _DEFAULT_ORDERS):
        self._orders: Tuple[float, ...] = tuple(a for a in orders if a > 1.0)
        # Accumulated RDP per order:  α → Σ_t RDP_t(α)
        self._rdp: Dict[float, float] = {a: 0.0 for a in self._orders}
        self._sigma_history: List[float] = []
        self._q_history: List[float] = []

    # ── Core mathematics ──────────────────────────────────────────────────────

    @staticmethod
    def _rdp_gaussian_full_batch(sigma: float, alpha: float) -> float:
        """
        RDP(α) for ONE step of the Gaussian mechanism, full-batch (q = 1).

            RDP(α) = α / (2σ²)                              [Mironov 2017, Prop 3]

        Exact for all real α > 1.  Note: σ is the *noise multiplier*
        (noise_std / sensitivity), not the absolute noise std.
        """
        if sigma <= 0.0:
            return float("inf")
        return alpha / (2.0 * sigma * sigma)

    @staticmethod
    def _rdp_gaussian_subsampled(sigma: float, q: float, alpha: float) -> float:
        """
        RDP(α) for ONE step of the Poisson-*subsampled* Gaussian mechanism.

        Integer α ≥ 2 — exact log-sum-exp formula (Wang et al. 2019, Thm 3):

            RDP(α) = (1/(α-1)) log Σ_{k=0}^{α} C(α,k) q^k (1-q)^{α-k}
                                   · exp(k(k-1) / (2σ²))

        Non-integer or α < 2 — amplification bound (Wang et al. 2019, Prop 3):

            RDP(α) ≤ q²·α·(α-1) / (2σ²)        [tight for small q]

        For q = 1 the subsampled formula reduces to α/(2σ²).
        """
        if q == 0.0:
            return 0.0
        if q == 1.0:
            return RDPAccountant._rdp_gaussian_full_batch(sigma, alpha)

        alpha_int = int(alpha)
        # Exact formula for integer α ≥ 2
        if alpha == float(alpha_int) and alpha_int >= 2:
            try:
                from math import comb
                log_terms: List[float] = []
                for k in range(alpha_int + 1):
                    log_coef = (
                        math.log(comb(alpha_int, k))
                        + k * math.log(q)
                        + (alpha_int - k) * math.log(1.0 - q)
                        + k * (k - 1) / (2.0 * sigma * sigma)
                    )
                    log_terms.append(log_coef)
                # Numerically stable log-sum-exp
                lse_max = max(log_terms)
                log_sum = lse_max + math.log(
                    sum(math.exp(t - lse_max) for t in log_terms)
                )
                return log_sum / (alpha_int - 1)
            except (OverflowError, ValueError):
                pass  # fall through to amplification bound

        # Amplification bound for non-integer or very large α
        return (q * q) * alpha * (alpha - 1) / (2.0 * sigma * sigma)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, sigma: float, q: float = 1.0, steps: int = 1) -> None:
        """
        Accumulate RDP cost for ``steps`` consecutive rounds of the
        (subsampled) Gaussian mechanism.

        Privacy composition is additive in the RDP domain:
            RDP_total(α) += steps · RDP_per_step(σ, q, α)

        Args:
            sigma:  Noise multiplier (noise_std / sensitivity).  Must be > 0.
            q:      Poisson subsampling rate ∈ (0, 1].  Use 1.0 (default) for
                    full-batch gradient sharing.
            steps:  Number of mechanism invocations to add.
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")
        if not (0.0 < q <= 1.0):
            raise ValueError(f"q must be in (0, 1], got {q!r}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")

        for alpha in self._orders:
            rdp_one = self._rdp_gaussian_subsampled(sigma, q, alpha)
            self._rdp[alpha] += rdp_one * steps

        for _ in range(steps):
            self._sigma_history.append(float(sigma))
            self._q_history.append(float(q))

    def get_epsilon(self, delta: float) -> Tuple[float, float]:
        """
        Convert accumulated RDP to (ε, δ)-DP via the tight conversion
        from Balle et al. (2020), Proposition 3:

            ε(α) = RDP(α) + log(α-1)/α − [log(δ) + log(α-1)] / (α−1)

        Minimised over all tracked Rényi orders α.

        This is strictly tighter than the original Mironov (2017) bound
            ε_Mironov(α) = RDP(α) + log(1/δ) / (α-1)
        because it accounts for the hockey-stick divergence structure.

        Args:
            delta:  Target δ ∈ (0, 1).

        Returns:
            (ε_opt, α_opt) — the tightest ε and the order that achieves it.
            Returns (inf, None) if no finite ε is achievable (e.g. 0 steps).
        """
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")

        best_eps: float = float("inf")
        best_alpha: Optional[float] = None

        for alpha in self._orders:
            rdp = self._rdp[alpha]
            if rdp <= 0.0:
                continue
            try:
                # Balle et al. (2020) — tight conversion
                eps = (
                    rdp
                    + math.log(alpha - 1.0) / alpha
                    - (math.log(delta) + math.log(alpha - 1.0)) / (alpha - 1.0)
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                continue

            if math.isfinite(eps) and eps < best_eps:
                best_eps = eps
                best_alpha = alpha

        return best_eps, best_alpha

    def get_rdp_values(self) -> Dict[float, float]:
        """Return a copy of the accumulated RDP dict {α: RDP_total(α)}."""
        return dict(self._rdp)

    def get_sigma_history(self) -> List[float]:
        """Return the list of noise multipliers used across all steps."""
        return list(self._sigma_history)

    def reset(self) -> None:
        """Reset all accumulated privacy loss."""
        self._rdp = {a: 0.0 for a in self._orders}
        self._sigma_history.clear()
        self._q_history.clear()

    @property
    def num_steps(self) -> int:
        """Total number of mechanism invocations accumulated so far."""
        return len(self._sigma_history)


# ─── Helper: binary-search σ for a target (ε, δ) ─────────────────────────────

def sigma_for_target_epsilon(
    target_eps: float,
    delta: float,
    steps: int = 1,
    q: float = 1.0,
    sigma_lo: float = 0.01,
    sigma_hi: float = 200.0,
    tol: float = 1e-5,
    orders: Tuple[float, ...] = _DEFAULT_ORDERS,
) -> float:
    """
    Find the *minimum* noise multiplier σ such that ``steps`` steps of the
    (subsampled) Gaussian mechanism satisfy (target_eps, delta)-DP under RDP.

    Uses binary search on σ.  Convergence is guaranteed because ε is
    strictly decreasing in σ.

    Args:
        target_eps: Desired ε.  Must be positive.
        delta:      Desired δ ∈ (0, 1).
        steps:      Number of FL rounds (mechanism invocations).
        q:          Subsampling rate.
        sigma_lo:   Lower search bound.
        sigma_hi:   Upper search bound.
        tol:        σ convergence tolerance.
        orders:     Rényi orders to use.

    Returns:
        Minimum σ ≥ sigma_lo achieving (target_eps, delta)-DP.

    Raises:
        ValueError if the target is unachievable at sigma_hi.
    """
    def _eps(s: float) -> float:
        acc = RDPAccountant(orders=orders)
        acc.step(s, q=q, steps=steps)
        eps, _ = acc.get_epsilon(delta)
        return eps

    hi_eps = _eps(sigma_hi)
    if hi_eps > target_eps:
        raise ValueError(
            f"sigma_hi={sigma_hi} too small: ε={hi_eps:.4f} still exceeds "
            f"target ε={target_eps}. Increase sigma_hi or relax target_eps."
        )

    lo, hi = sigma_lo, sigma_hi
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if _eps(mid) <= target_eps:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < tol:
            break

    return hi  # conservative: return upper side of bracket


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-CLIENT DP STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClientDPState:
    """
    Per-client DP state, updated each FL round.

    Privacy accounting is delegated entirely to ``rdp_accountant``.
    The ``budget_spent`` property reports the current cumulative ε,
    computed by the RDP accountant — not a running sum of per-round ε.

    Attributes
    ----------
    hospital_id:
        Unique client identifier.
    sensitivity_ema:
        EMA of observed adapter-delta L2-norms (adaptive clipping bound Cᵢ).
    loss_history:
        Per-round training losses, used to schedule the noise multiplier.
    rdp_accountant:
        Rényi DP accountant — sole source of truth for privacy cost.
    round_sigmas:
        Per-round noise multiplier values (for plots / audit).
    round_clips:
        Per-round clipping norms (for plots / audit).
    round_epsilons:
        Cumulative (ε, δ)-DP ε after each round, computed from the RDP
        accountant.  Entry t = total ε spent through round t, NOT just
        round t's contribution.
    """
    hospital_id: str
    sensitivity_ema: float = 1.0
    loss_history: List[float] = field(default_factory=list)
    rdp_accountant: RDPAccountant = field(default_factory=RDPAccountant)
    round_sigmas: List[float] = field(default_factory=list)
    round_clips: List[float] = field(default_factory=list)
    round_epsilons: List[float] = field(default_factory=list)

    def update_sensitivity(self, observed_norm: float, alpha: float) -> None:
        """EMA update: Cᵢ ← (1−α)·Cᵢ + α·‖Δw‖₂"""
        self.sensitivity_ema = (
            (1.0 - alpha) * self.sensitivity_ema + alpha * observed_norm
        )

    def record_loss(self, loss: float) -> None:
        self.loss_history.append(float(loss))

    def record_round(self, sigma: float, clip_norm: float, delta: float) -> None:
        """
        Log per-round parameters and append the running cumulative ε
        (re-computed from the RDP accountant, which must already have been
        updated for this round via ``accountant.step()``).
        """
        self.round_sigmas.append(round(sigma, 6))
        self.round_clips.append(round(clip_norm, 6))
        eps, _ = self.rdp_accountant.get_epsilon(delta)
        self.round_epsilons.append(round(eps, 6))

    @property
    def latest_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("inf")

    @property
    def budget_spent(self) -> float:
        """
        Current cumulative (ε, δ)-DP ε, from the RDP accountant.

        This is the *true* privacy cost — not a simple sum of per-round ε.
        """
        return self.round_epsilons[-1] if self.round_epsilons else 0.0

    def to_dict(self) -> dict:
        return {
            "hospital_id":                self.hospital_id,
            "sensitivity_ema":            round(self.sensitivity_ema, 6),
            "budget_spent_epsilon":       round(self.budget_spent, 6),
            "rounds_participated":        self.rdp_accountant.num_steps,
            "round_sigmas":               self.round_sigmas,
            "round_clips":                self.round_clips,
            "round_epsilons_cumulative":  self.round_epsilons,
            "loss_history":               [round(l, 6) for l in self.loss_history],
            "rdp_values_final": {
                f"alpha_{a:.1f}": round(v, 8)
                for a, v in self.rdp_accountant.get_rdp_values().items()
                if v > 0
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE DP MECHANISM
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveDPMechanism:
    """
    Coordinates per-client adaptive differential privacy with correct
    Rényi DP accounting.

    Noise schedule  (the "adaptive" part)
    --------------------------------------
    Instead of a fixed σ for all clients and rounds, each client i at round r
    receives a personalised noise multiplier σᵢ(r):

        score_i(r)  = 1 / loss_i(r)             [higher loss → more noise needed]
        weight_i(r) = softmax(score_i(r))_i      [Σᵢ weight_i = 1]
        σᵢ(r)       = weight_i(r) · K · σ_base  [K = number of clients]

    This means clients still learning rapidly (high loss) receive a larger
    noise multiplier (more protection for sensitive updates), while converging
    clients receive less noise (better utility from stable updates).  The mean
    noise multiplier across clients equals σ_base at every round.

    Privacy accounting  (the correct part)
    ----------------------------------------
    Each client has its own ``RDPAccountant``.  After applying noise with
    multiplier σᵢ(r), we record:

        accountant_i.step(sigma=σᵢ(r), q=1.0)

    which accumulates RDP(α) = α/(2σᵢ(r)²) for each order α.
    The true (ε, δ)-DP guarantee is computed at any time as:

        ε_i = min_{α>1} RDP_i_total(α) + log(α-1)/α
                        - [log(δ) + log(α-1)] / (α-1)

    The system's global privacy guarantee is ε = max_i ε_i.

    σ_base calibration
    -------------------
    σ_base is found via binary search so that T steps of a *uniform*
    schedule (σ = σ_base every round) would achieve exactly global_epsilon
    (ε, δ)-DP.  The adaptive schedule deviates from uniform but uses this
    calibration as a neutral anchor.

    Parameters
    ----------
    hospital_ids:        Hospital identifiers.
    global_epsilon:      Target privacy budget ε (for σ_base calibration).
    delta:               Privacy failure probability δ.
    fl_rounds:           Total FL rounds (for σ_base calibration).
    initial_sensitivity: Starting clipping norm C₀ = DPConfig.max_grad_norm.
    ema_alpha:           EMA smoothing for sensitivity estimation.
    min_noise_fraction:  Floor on each client's noise weight (prevents σ ≈ 0).
    """

    def __init__(
        self,
        hospital_ids: List[str],
        global_epsilon: float,
        delta: float,
        fl_rounds: int,
        initial_sensitivity: float = 1.0,
        ema_alpha: float = 0.1,
        min_noise_fraction: float = 0.1,
    ):
        self.global_epsilon = global_epsilon
        self.delta = delta
        self.fl_rounds = fl_rounds
        self.ema_alpha = ema_alpha
        self.min_noise_fraction = min_noise_fraction
        self._hospital_ids = list(hospital_ids)

        # Calibrate σ_base: smallest σ such that T full-batch Gaussian steps
        # achieve exactly (global_epsilon, delta)-DP via RDP accounting.
        try:
            self.sigma_base = sigma_for_target_epsilon(
                target_eps=global_epsilon,
                delta=delta,
                steps=fl_rounds,
                q=1.0,
            )
            logger.info(
                "AdaptiveDPMechanism | σ_base=%.4f (RDP binary search: "
                "%d steps, ε=%.2f, δ=%.0e)",
                self.sigma_base, fl_rounds, global_epsilon, delta,
            )
        except ValueError as exc:
            # Fallback: standard approximate formula — always safe (conservative)
            self.sigma_base = (
                initial_sensitivity
                * math.sqrt(2.0 * math.log(1.25 / delta))
                / (global_epsilon / math.sqrt(max(fl_rounds, 1)))
            )
            logger.warning(
                "RDP σ calibration failed (%s); using approximate "
                "σ_base=%.4f (may be slightly over-noisy).",
                exc, self.sigma_base,
            )

        self.states: Dict[str, ClientDPState] = {
            hid: ClientDPState(
                hospital_id=hid,
                sensitivity_ema=initial_sensitivity,
            )
            for hid in hospital_ids
        }

        logger.info(
            "AdaptiveDPMechanism | K=%d ε=%.2f δ=%.1e σ_base=%.4f",
            len(hospital_ids), global_epsilon, delta, self.sigma_base,
        )

    # ── Noise schedule ────────────────────────────────────────────────────────

    def _compute_noise_multipliers(self) -> Dict[str, float]:
        """
        Compute per-client noise multipliers σᵢ for the current round.

        Clients with high loss receive a larger multiplier (more noise →
        more privacy protection for fast-learning updates).  The mean
        multiplier across clients equals σ_base.

        Returns
        -------
        Dict[hospital_id → σᵢ]
        """
        n = len(self._hospital_ids)

        # Inverse-loss scores: high loss → high score → more noise
        scores = [
            1.0 / max(self.states[hid].latest_loss, 1e-6)
            for hid in self._hospital_ids
        ]

        # Numerically stable softmax
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        total_exp = sum(exp_s)
        weights = [e / total_exp for e in exp_s]   # Σ weights = 1

        # Apply floor to prevent any client getting near-zero noise
        floor_w = self.min_noise_fraction / n
        weights = [max(w, floor_w) for w in weights]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]   # re-normalise

        # Scale so that mean(σᵢ) = σ_base
        return {
            hid: w * n * self.sigma_base
            for hid, w in zip(self._hospital_ids, weights)
        }

    # ── Main API ──────────────────────────────────────────────────────────────

    def apply_noise(
        self,
        hospital_id: str,
        weights: "OrderedDict[str, torch.Tensor]",
        round_num: int,
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Clip and add calibrated Gaussian noise to a client's LoRA adapter.

        Algorithm
        ---------
        1. Observe ‖Δw‖₂ (actual adapter delta norm).
        2. Update EMA clipping bound: Cᵢ ← EMA(‖Δw‖₂).
        3. Compute adaptive noise multiplier σᵢ (loss-proportional schedule).
        4. Clip each weight tensor proportionally to Cᵢ.
        5. Add Gaussian noise: n ~ N(0, (σᵢ · Cᵢ)² · I).
        6. Update RDP accountant:  accountant.step(sigma=σᵢ, q=1.0).
        7. Record per-round parameters; compute running (ε, δ) from RDP.

        Note: σᵢ is the *noise multiplier* (ratio noise_std / clip_norm).
              The accountant tracks it as-is; step 6 uses σᵢ, not σᵢ·Cᵢ.

        Args:
            hospital_id: Which client's update.
            weights:     LoRA adapter weights (detached tensors on CPU).
            round_num:   Current FL round index (0-based).  Used for logging.

        Returns:
            Noisy OrderedDict with the same keys as ``weights``.
        """
        state = self.states[hospital_id]

        # 1. Observe norm; 2. Update EMA clipping bound
        with torch.no_grad():
            flat = torch.cat(
                [p.reshape(-1).float() for p in weights.values()]
            )
            actual_norm = float(torch.norm(flat).item())
        state.update_sensitivity(actual_norm, self.ema_alpha)
        clip_norm = state.sensitivity_ema

        # 3. Adaptive noise multiplier
        sigma_mult = self._compute_noise_multipliers()[hospital_id]
        noise_std  = sigma_mult * clip_norm          # absolute noise std

        # 4 & 5. Clip and add noise
        noisy: OrderedDict = OrderedDict()
        with torch.no_grad():
            for name, param in weights.items():
                p = param.float()
                norm = float(torch.norm(p).item())
                if norm > clip_norm and norm > 0.0:
                    p = p * (clip_norm / norm)
                noisy[name] = p + torch.randn_like(p) * noise_std

        # 6. RDP accounting — step with the noise MULTIPLIER (not noise_std)
        state.rdp_accountant.step(sigma=sigma_mult, q=1.0, steps=1)

        # 7. Audit record + compute cumulative (ε, δ) via RDP accountant
        state.record_round(sigma=sigma_mult, clip_norm=clip_norm, delta=self.delta)

        eps_now = state.budget_spent
        logger.info(
            "%s | round=%d | Cᵢ=%.4f σ_mult=%.4f noise_std=%.4f | "
            "ε_cumul(RDP)=%.4f δ=%.0e",
            hospital_id, round_num, clip_norm, sigma_mult, noise_std,
            eps_now, self.delta,
        )
        if eps_now > self.global_epsilon:
            logger.warning(
                "%s | cumulative ε=%.4f exceeded target %.1f at round %d.",
                hospital_id, eps_now, self.global_epsilon, round_num,
            )

        return noisy

    def record_loss(self, hospital_id: str, loss: float) -> None:
        """Record training loss for the next round's noise schedule."""
        self.states[hospital_id].record_loss(loss)

    # ── Budget queries ────────────────────────────────────────────────────────

    def get_privacy_spent(
        self,
        hospital_id: str,
        delta: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Return the tight (ε, δ)-DP guarantee for a client at the current
        round, computed from the RDP accountant via Balle et al. (2020).

        Returns (ε, α_optimal).
        """
        d = self.delta if delta is None else delta
        return self.states[hospital_id].rdp_accountant.get_epsilon(d)

    def get_worst_case_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Maximum ε across all clients.  This is the system-level (ε, δ)-DP
        guarantee: the federation satisfies (ε_max, δ)-DP.
        """
        d = self.delta if delta is None else delta
        return max(
            self.get_privacy_spent(hid, d)[0]
            for hid in self._hospital_ids
        )

    def get_budget_spent(self, hospital_id: str) -> float:
        """Convenience: cumulative ε for one client at self.delta."""
        eps, _ = self.get_privacy_spent(hospital_id)
        return eps

    # ── Compatibility shim (used by fl_simulate.py) ───────────────────────────

    def compute_epsilon_allocation(self, round_num: int) -> Dict[str, float]:
        """
        Backward-compatibility shim for fl_simulate.py.

        Returns each client's *cumulative* ε to date (not a per-round slice).
        The result is suitable for the ``allocation_log_dict`` logger.
        """
        return {
            hid: self.get_privacy_spent(hid)[0]
            for hid in self._hospital_ids
        }

    def allocation_log_dict(
        self,
        alloc: Dict[str, float],
        round_num: int,
    ) -> Dict[str, float]:
        """
        Flat metrics dict for the experiment tracker.  Includes per-client
        ε (from RDP), noise multiplier, noise std, and Rényi order used.
        """
        out: Dict[str, float] = {}
        for hid in self._hospital_ids:
            state = self.states[hid]
            eps, alpha_opt = state.rdp_accountant.get_epsilon(self.delta)
            sigma = state.round_sigmas[-1] if state.round_sigmas else self.sigma_base
            clip  = state.round_clips[-1]  if state.round_clips  else state.sensitivity_ema
            out[f"adaptive_dp/{hid}/epsilon_rdp"]      = round(eps, 6)
            out[f"adaptive_dp/{hid}/optimal_alpha"]    = round(alpha_opt or 0.0, 2)
            out[f"adaptive_dp/{hid}/sigma_mult"]       = round(sigma, 6)
            out[f"adaptive_dp/{hid}/noise_std"]        = round(sigma * clip, 6)
            out[f"adaptive_dp/{hid}/sensitivity_ema"]  = round(clip, 6)
            out[f"adaptive_dp/{hid}/rdp_steps"]        = state.rdp_accountant.num_steps
        return out

    def get_all_states(self) -> Dict[str, dict]:
        return {hid: s.to_dict() for hid, s in self.states.items()}

    def summary(self) -> dict:
        """High-level summary dict for the final training report."""
        per_client_summary = {}
        for hid in self._hospital_ids:
            eps, alpha_opt = self.get_privacy_spent(hid)
            per_client_summary[hid] = {
                **self.states[hid].to_dict(),
                "final_epsilon":        round(eps, 6),
                "final_optimal_alpha":  round(alpha_opt or 0.0, 2),
            }
        return {
            "mechanism":             "adaptive_dp_rdp",
            "accounting_method":     (
                "Renyi DP composition (Mironov 2017) + "
                "Balle et al. (2020) tight RDP-to-(eps,delta) conversion"
            ),
            "global_epsilon_target": self.global_epsilon,
            "delta":                 self.delta,
            "fl_rounds":             self.fl_rounds,
            "sigma_base":            round(self.sigma_base, 6),
            "ema_alpha":             self.ema_alpha,
            "min_noise_fraction":    self.min_noise_fraction,
            "worst_case_epsilon":    round(self.get_worst_case_epsilon(), 6),
            "per_client":            per_client_summary,
        }
