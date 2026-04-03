"""
MedTrace — Adaptive Differential Privacy Mechanism
===================================================
Implements per-client noise calibration for federated learning.

Algorithm overview
------------------
Standard federated DP uses a single global noise level σ for all clients and
all rounds.  This is wasteful: a hospital whose local model has already
converged adds the same noise as one still learning rapidly, and a hospital
with tightly clustered gradients (low sensitivity) gets the same clipping as
one with high-variance updates.

Adaptive DP (this module) maintains per-client state that evolves each round:

1.  **Gradient sensitivity estimation (adaptive clipping)**
    Each client tracks an exponential moving average (EMA) of its observed
    LoRA gradient norms.  The EMA replaces the fixed global clipping norm C
    with a client-specific Cᵢ(t), tightening the bound as gradients shrink
    during convergence.

2.  **Loss-proportional epsilon allocation**
    The global privacy budget ε is redistributed each round across clients
    in proportion to their *inverse training loss* via softmax:

        weight_i(t) = softmax(1 / loss_i(t))

    Clients with low loss (already converging) receive a larger ε fraction —
    less noise, more informative updates.  Clients still learning rapidly
    receive a smaller ε slice — more noise protects their sensitive updates.

    A configurable floor ``min_epsilon_fraction`` ensures no client is
    starved to ε ≈ 0.

3.  **Advanced-composition budget tracking**
    Per-round per-client spend:  εᵢ(t) = weight_i(t) · ε_global / √T
    Cumulative spend is tracked per client and capped at global_epsilon.

References
----------
* McMahan et al. (2018) — Learning Differentially Private Recurrent Language
  Models (Gaussian mechanism baseline for FL)
* Geyer et al. (2017) — Differentially Private Federated Learning: A Client
  Level Perspective (per-client DP motivation)
* Andrew et al. (2021) — Differentially Private Learning with Adaptive
  Clipping (adaptive clipping inspiration)
* Yu et al. (2021) — Differentially Private Fine-Tuning of Language Models
  (DP + LoRA in LLM context)
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ─── Per-Client State ────────────────────────────────────────────────────────

@dataclass
class ClientDPState:
    """
    Mutable per-client DP state, updated each FL round.

    Attributes
    ----------
    hospital_id:
        Unique identifier for the hospital node.
    sensitivity_ema:
        Exponential moving average of observed LoRA gradient L2-norms.
        This is the adaptive clipping bound Cᵢ(t).
    loss_history:
        Per-round training losses used to compute ε allocation weights.
    budget_spent:
        Cumulative ε consumed by this client (advanced composition).
    round_epsilons:
        Per-round ε values allocated to this client (for audit/plotting).
    round_sigmas:
        Per-round Gaussian σ values (for audit/plotting).
    round_sensitivities:
        Per-round Cᵢ values (adaptive clipping norms).
    """
    hospital_id: str
    sensitivity_ema: float = 1.0
    loss_history: List[float] = field(default_factory=list)
    budget_spent: float = 0.0
    round_epsilons: List[float] = field(default_factory=list)
    round_sigmas: List[float] = field(default_factory=list)
    round_sensitivities: List[float] = field(default_factory=list)

    # ── Mutators ───────────────────────────────────────────────────────────

    def update_sensitivity(self, observed_norm: float, alpha: float) -> None:
        """EMA update: Cᵢ(t+1) = (1−α)·Cᵢ(t) + α·‖g‖₂"""
        self.sensitivity_ema = (1.0 - alpha) * self.sensitivity_ema + alpha * observed_norm

    def record_loss(self, loss: float) -> None:
        """Append training loss for use in next round's ε allocation."""
        self.loss_history.append(float(loss))

    def record_round(self, eps: float, sigma: float, clip_norm: float) -> None:
        """Store per-round DP parameters for audit trail."""
        self.round_epsilons.append(round(eps, 6))
        self.round_sigmas.append(round(sigma, 6))
        self.round_sensitivities.append(round(clip_norm, 6))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def latest_loss(self) -> float:
        """Most recent training loss, or inf if no rounds completed yet."""
        return self.loss_history[-1] if self.loss_history else float("inf")

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "hospital_id": self.hospital_id,
            "sensitivity_ema": round(self.sensitivity_ema, 6),
            "budget_spent": round(self.budget_spent, 6),
            "round_epsilons": self.round_epsilons,
            "round_sigmas": self.round_sigmas,
            "round_sensitivities": self.round_sensitivities,
            "loss_history": [round(l, 6) for l in self.loss_history],
        }


# ─── Adaptive Mechanism ──────────────────────────────────────────────────────

class AdaptiveDPMechanism:
    """
    Coordinates per-client adaptive differential privacy for all hospital
    nodes in one FL experiment.

    The mechanism is the single authoritative source for:
      - ε allocation per client per round
      - Adaptive clipping norm (EMA of gradient norms)
      - Gaussian noise application
      - Budget accounting

    ``fl_simulate.py`` creates one instance, passes it to each
    ``HospitalClient.train_local()`` call, and queries it for the tracker.

    Parameters
    ----------
    hospital_ids:
        List of hospital identifiers (same order as FLConfig.hospitals).
    global_epsilon:
        Total privacy budget ε shared across all clients.
    delta:
        DP failure probability δ (same for all clients).
    fl_rounds:
        Total number of FL rounds (used in advanced composition denominator).
    initial_sensitivity:
        Starting clipping norm Cᵢ for all clients.  Should equal
        ``DPConfig.max_grad_norm`` so round 0 behaves like standard DP.
    ema_alpha:
        EMA smoothing factor α for sensitivity updates (0 < α ≤ 1).
        Smaller → more stable, slower to adapt.
    min_epsilon_fraction:
        Floor on each client's ε weight before normalisation.
        Value 0.1 means every client receives at least 10 % of the mean ε.
    """

    def __init__(
        self,
        hospital_ids: List[str],
        global_epsilon: float,
        delta: float,
        fl_rounds: int,
        initial_sensitivity: float = 1.0,
        ema_alpha: float = 0.1,
        min_epsilon_fraction: float = 0.1,
    ):
        self.global_epsilon = global_epsilon
        self.delta = delta
        self.fl_rounds = fl_rounds
        self.ema_alpha = ema_alpha
        self.min_epsilon_fraction = min_epsilon_fraction

        self.states: Dict[str, ClientDPState] = {
            hid: ClientDPState(hospital_id=hid, sensitivity_ema=initial_sensitivity)
            for hid in hospital_ids
        }

        logger.info(
            "AdaptiveDPMechanism | hospitals=%d ε=%.1f δ=%.1e α=%.2f floor=%.2f",
            len(hospital_ids), global_epsilon, delta, ema_alpha, min_epsilon_fraction,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_epsilon_allocation(self, round_num: int) -> Dict[str, float]:
        """
        Compute per-client ε values for this round.

        Steps:
        1. Compute quality scores = 1 / loss_i  (higher loss → lower score)
        2. Softmax normalise to get allocation weights w_i  (sum = 1)
        3. Apply floor: w_i ← max(w_i, floor / N); re-normalise
        4. Per-round budget: εᵢ = w_i · N · (ε_global / √T)
        5. Cap at remaining budget for each client

        Returns
        -------
        Dict mapping hospital_id → ε for this round.
        """
        hospital_ids = list(self.states.keys())
        n = len(hospital_ids)

        # Step 1: quality scores (inverse loss)
        scores = [
            1.0 / max(self.states[hid].latest_loss, 1e-6)
            for hid in hospital_ids
        ]

        # Step 2: numerically stable softmax
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        total_exp = sum(exp_s)
        weights = [e / total_exp for e in exp_s]  # sum = 1.0

        # Step 3: apply floor and re-normalise
        floor_w = self.min_epsilon_fraction / n
        weights = [max(w, floor_w) for w in weights]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Step 4: per-round allocation
        # Advanced composition: ε_global = per_round_eps * √T
        # So per_round_base = ε_global / √T
        per_round_base = self.global_epsilon / math.sqrt(max(self.fl_rounds, 1))
        alloc = {
            hid: w * n * per_round_base
            for hid, w in zip(hospital_ids, weights)
        }

        # Step 5: cap at remaining budget
        alloc = {
            hid: max(
                min(eps, self.global_epsilon - self.states[hid].budget_spent),
                0.0,
            )
            for hid, eps in alloc.items()
        }

        logger.debug(
            "Round %d | ε allocation: %s",
            round_num,
            {hid: f"{eps:.4f}" for hid, eps in alloc.items()},
        )
        return alloc

    def apply_noise(
        self,
        hospital_id: str,
        weights: "OrderedDict[str, torch.Tensor]",
        round_num: int,
        round_epsilon: float,
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Apply adaptive Gaussian DP noise to a client's LoRA weight update.

        1. Compute observed gradient norm (for EMA update).
        2. Update sensitivity EMA.
        3. Clip each weight tensor by Cᵢ (adaptive bound).
        4. Add Gaussian noise calibrated to (εᵢ, δ) with scale Cᵢ.
        5. Update budget accounting.

        Args:
            hospital_id: which hospital's update this is.
            weights: LoRA adapter weights (detached, on CPU).
            round_num: current FL round (0-based), used for logging.
            round_epsilon: ε allocated to this client this round.

        Returns:
            Noisy OrderedDict with same keys as input.
        """
        state = self.states[hospital_id]

        if round_epsilon <= 0.0:
            logger.warning(
                "%s round %d: round_epsilon=%.6f, privacy budget exhausted — "
                "returning unnoised weights (budget fully spent).",
                hospital_id, round_num, round_epsilon,
            )
            return weights

        clip_norm = state.sensitivity_ema

        # Compute σ for Gaussian mechanism: σ = C · √(2 ln(1.25/δ)) / ε
        sigma = clip_norm * math.sqrt(2.0 * math.log(1.25 / self.delta)) / round_epsilon

        # Observe actual norm of the full weight update (for EMA)
        with torch.no_grad():
            flat = torch.cat([p.reshape(-1).float() for p in weights.values()])
            actual_norm = torch.norm(flat).item()

        # Update EMA sensitivity before applying noise
        state.update_sensitivity(actual_norm, self.ema_alpha)

        # Apply noise per tensor
        noisy: OrderedDict = OrderedDict()
        with torch.no_grad():
            for name, param in weights.items():
                p = param.float()
                # Per-tensor proportional clipping
                norm = torch.norm(p).item()
                if norm > clip_norm and norm > 0:
                    p = p * (clip_norm / norm)
                noisy[name] = p + torch.randn_like(p) * sigma

        # Update accounting
        state.budget_spent += round_epsilon
        state.record_round(round_epsilon, sigma, clip_norm)

        logger.info(
            "%s | adaptive DP | Cᵢ=%.4f σ=%.4f ε_round=%.4f ε_spent=%.3f/%.1f",
            hospital_id, clip_norm, sigma, round_epsilon,
            state.budget_spent, self.global_epsilon,
        )
        return noisy

    def record_loss(self, hospital_id: str, loss: float) -> None:
        """
        Record a hospital's training loss for the *next* round's ε allocation.
        Must be called after train_local() returns and before the next round's
        compute_epsilon_allocation().
        """
        self.states[hospital_id].record_loss(loss)

    def get_budget_spent(self, hospital_id: str) -> float:
        """Return cumulative ε spent by a hospital."""
        return self.states[hospital_id].budget_spent

    def get_all_states(self) -> Dict[str, dict]:
        """Serialise all client DP states (for checkpointing / logging)."""
        return {hid: s.to_dict() for hid, s in self.states.items()}

    def allocation_log_dict(
        self, alloc: Dict[str, float], round_num: int
    ) -> Dict[str, float]:
        """
        Return a flat dict of tracker-ready metrics from an allocation map.
        Includes per-client ε, σ, and sensitivity for the tracker.
        """
        out: Dict[str, float] = {}
        for hid, eps in alloc.items():
            state = self.states[hid]
            clip = state.sensitivity_ema
            if eps > 0:
                sigma = clip * math.sqrt(2.0 * math.log(1.25 / self.delta)) / eps
            else:
                sigma = float("inf")
            out[f"adaptive_dp/{hid}/round_epsilon"] = eps
            out[f"adaptive_dp/{hid}/sigma"] = sigma
            out[f"adaptive_dp/{hid}/sensitivity_ema"] = clip
            out[f"adaptive_dp/{hid}/budget_spent"] = state.budget_spent
        return out

    def summary(self) -> dict:
        """High-level summary dict for the final training report."""
        return {
            "mechanism": "adaptive_dp",
            "global_epsilon": self.global_epsilon,
            "delta": self.delta,
            "fl_rounds": self.fl_rounds,
            "ema_alpha": self.ema_alpha,
            "min_epsilon_fraction": self.min_epsilon_fraction,
            "per_client": self.get_all_states(),
        }
