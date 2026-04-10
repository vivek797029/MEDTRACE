"""
MedTrace — Trust-Weighted Federated Aggregation (TrustFedAvg)
=============================================================
Novel contribution: replaces sample-count-only FedAvg weighting with
a multi-dimensional trust score that detects gradient misalignment,
anomalous update norms, and slow-converging clients.

Algorithm (TrustFedAvg)
-----------------------
Standard FedAvg assigns each client a weight proportional to its
data-set size:

    w_i = n_i / Σ n_j

This is unaware of update *quality*.  A client whose gradients point
in the wrong direction (data poisoning, distribution shift, or
transient training instability) still receives full weight.

TrustFedAvg augments the data-fraction weight with three independent
trust signals computed entirely from the clients' transmitted update
tensors — requiring no additional communication:

1.  **Cosine Alignment Score (CAS)**
    Measures directional agreement between client i's update Δwᵢ and
    the element-wise mean of all updates  Δw̄:

        cas_i = (cosine(Δwᵢ, Δw̄) + 1) / 2   ∈ [0, 1]

    Clients whose updates oppose the consensus direction receive
    low CAS and are down-weighted without being discarded.

2.  **Loss-Convergence Score (LCS)**
    Rewards clients that have achieved lower training loss:

        lcs_i = softmax( 1 / (loss_i + ε) )_i

    A hospital that has already converged contributes high-quality,
    low-noise updates and deserves higher weight.  A hospital still
    oscillating with high loss is penalised to limit its influence.

3.  **Norm Consistency Score (NCS)**
    Flags outlier update magnitudes via the Median Absolute Deviation
    (MAD) of the per-client L2 norms, inspired by coordinate-wise
    median defences (Yin et al., 2018):

        ncs_i = exp( -|‖Δwᵢ‖ − med_norm| / (MAD + ε) )

    Clients with abnormally large or small update norms (e.g. gradient
    explosion, under-fitting, or Byzantine injection) receive low NCS.

Combined trust score (configurable weights α₁, α₂, α₃, sum = 1):

    T_i = α₁ · cas_i + α₂ · lcs_i + α₃ · ncs_i

Final aggregation weight (data-size modulated):

    w_i = softmax(T_i) · n_i
          ─────────────────────
          Σ  softmax(T_j) · n_j

Graceful degradation
--------------------
On round 0, loss histories are empty and cosine scores cannot yet be
computed against a rolling mean (no prior global direction).  In that
round TrustFedAvg falls back to standard FedAvg; trust scoring
activates from round 1 onward.

References
----------
* McMahan et al. (2017) — Communication-Efficient Learning of Deep
  Networks from Decentralized Data  (FedAvg baseline)
* Yin et al. (2018) — Byzantine-Robust Distributed Learning: Towards
  Optimal Statistical Rates  (median-based Byzantine defence)
* Blanchard et al. (2017) — Machine Learning with Adversaries: Byzantine
  Tolerant Gradient Descent  (Krum inspiration)
* Fang et al. (2020) — Local Model Poisoning Attacks to Byzantine-Robust
  FL  (trust-scoring motivation)
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# ─── Per-Client Trust State ──────────────────────────────────────────────────

@dataclass
class ClientTrustState:
    """
    Mutable trust accounting state for one hospital node.

    Attributes
    ----------
    hospital_id:
        Unique node identifier.
    loss_history:
        Per-round training losses used to compute LCS.
    trust_history:
        Final per-round trust scores T_i (for audit/plotting).
    weight_history:
        Final per-round aggregation weights w_i (post data-size modulation).
    cas_history:
        Per-round cosine alignment scores.
    lcs_history:
        Per-round loss-convergence scores.
    ncs_history:
        Per-round norm consistency scores.
    """
    hospital_id: str
    loss_history: List[float] = field(default_factory=list)
    trust_history: List[float] = field(default_factory=list)
    weight_history: List[float] = field(default_factory=list)
    cas_history: List[float] = field(default_factory=list)
    lcs_history: List[float] = field(default_factory=list)
    ncs_history: List[float] = field(default_factory=list)

    def record_round(
        self,
        trust: float,
        weight: float,
        cas: float,
        lcs: float,
        ncs: float,
    ) -> None:
        """Append per-round scores for audit trail."""
        self.trust_history.append(round(trust, 6))
        self.weight_history.append(round(weight, 6))
        self.cas_history.append(round(cas, 6))
        self.lcs_history.append(round(lcs, 6))
        self.ncs_history.append(round(ncs, 6))

    def record_loss(self, loss: float) -> None:
        self.loss_history.append(float(loss))

    @property
    def latest_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("inf")

    def to_dict(self) -> dict:
        return {
            "hospital_id": self.hospital_id,
            "loss_history": [round(l, 6) for l in self.loss_history],
            "trust_history": self.trust_history,
            "weight_history": self.weight_history,
            "cas_history": self.cas_history,
            "lcs_history": self.lcs_history,
            "ncs_history": self.ncs_history,
        }


# ─── Trust-Weighted Aggregator ───────────────────────────────────────────────

class TrustWeightedAggregator:
    """
    Computes per-round trust scores and trust-modulated aggregation weights
    for all hospital nodes.

    This class is the single authoritative source for TrustFedAvg weighting.
    ``FederatedServer.aggregate()`` calls ``compute_weights()`` when trust
    aggregation is enabled and uses the returned weights instead of the
    plain data-fraction weights of standard FedAvg.

    Parameters
    ----------
    hospital_ids:
        List of hospital identifiers, same order as FLConfig.hospitals.
    alpha_cas:
        Weight for cosine alignment score.
    alpha_lcs:
        Weight for loss-convergence score.
    alpha_ncs:
        Weight for norm consistency score.
        (alpha_cas + alpha_lcs + alpha_ncs should equal 1.0)
    trust_temperature:
        Softmax temperature τ applied to trust scores before mixing with
        data-fraction weights.  Higher τ → more uniform weights (closer to
        pure FedAvg); lower τ → sharper differentiation.
    min_trust:
        Hard floor on any single client's trust score before softmax.
        Prevents complete exclusion of a client from aggregation.
    """

    def __init__(
        self,
        hospital_ids: List[str],
        alpha_cas: float = 0.40,
        alpha_lcs: float = 0.35,
        alpha_ncs: float = 0.25,
        trust_temperature: float = 1.0,
        min_trust: float = 0.05,
    ):
        # Normalise weights to sum to 1 regardless of user input
        total = alpha_cas + alpha_lcs + alpha_ncs
        if total <= 0:
            raise ValueError("Trust component weights must sum to a positive value.")
        self.alpha_cas = alpha_cas / total
        self.alpha_lcs = alpha_lcs / total
        self.alpha_ncs = alpha_ncs / total
        self.trust_temperature = max(trust_temperature, 1e-6)
        self.min_trust = min_trust

        self.states: Dict[str, ClientTrustState] = {
            hid: ClientTrustState(hospital_id=hid)
            for hid in hospital_ids
        }
        # Rolling mean of all previous rounds' mean update for CAS fallback
        self._prev_mean_flat: Optional["torch.Tensor"] = None

        logger.info(
            "TrustWeightedAggregator | hospitals=%d "
            "α_cas=%.2f α_lcs=%.2f α_ncs=%.2f τ=%.2f floor=%.3f",
            len(hospital_ids),
            self.alpha_cas, self.alpha_lcs, self.alpha_ncs,
            self.trust_temperature, self.min_trust,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_weights(
        self,
        client_updates: Dict[str, Tuple["OrderedDict", dict]],
        round_num: int,
    ) -> Dict[str, float]:
        """
        Compute trust-modulated aggregation weights for one FL round.

        Parameters
        ----------
        client_updates:
            Mapping of hospital_id → (weight_dict, metrics_dict).
            metrics_dict must contain 'num_samples' and 'train_loss'.
        round_num:
            Current FL round (0-based).  Round 0 falls back to FedAvg.

        Returns
        -------
        Dict mapping hospital_id → float aggregation weight (sum = 1).
        """
        hids = list(client_updates.keys())
        n_clients = len(hids)

        if n_clients == 1:
            # Single client: trust score is irrelevant
            return {hids[0]: 1.0}

        # ── Extract weight tensors and metrics ────────────────────────────
        weight_dicts = {hid: client_updates[hid][0] for hid in hids}
        metrics = {hid: client_updates[hid][1] for hid in hids}
        sample_counts = {hid: metrics[hid]["num_samples"] for hid in hids}
        losses = {hid: metrics[hid].get("train_loss", float("inf")) for hid in hids}

        # Flatten each client's weight update into a 1-D tensor
        flat: Dict[str, "torch.Tensor"] = {}
        norms: Dict[str, float] = {}
        with torch.no_grad():
            for hid, wd in weight_dicts.items():
                f = torch.cat([p.reshape(-1).float() for p in wd.values()])
                flat[hid] = f
                norms[hid] = torch.norm(f).item()

        # ── Round 0: insufficient history → pure FedAvg ──────────────────
        if round_num == 0 or self._prev_mean_flat is None:
            logger.info(
                "Round %d: TrustFedAvg falling back to FedAvg "
                "(no prior round history for CAS computation)",
                round_num + 1,
            )
            total_n = sum(sample_counts.values())
            vanilla_weights = {
                hid: sample_counts[hid] / max(total_n, 1) for hid in hids
            }
            # Record uniform trust score placeholder
            for hid in hids:
                self.states[hid].record_round(
                    trust=1.0 / n_clients,
                    weight=vanilla_weights[hid],
                    cas=1.0 / n_clients,
                    lcs=1.0 / n_clients,
                    ncs=1.0 / n_clients,
                )
            # Store mean update for next round's CAS
            mean_flat = torch.stack(list(flat.values())).mean(dim=0)
            self._prev_mean_flat = mean_flat.detach().clone()
            return vanilla_weights

        # ── Component 1: Cosine Alignment Score ──────────────────────────
        # Reference direction: mean update from *previous* round
        # (using prev round's mean to avoid circular dependency within
        # this round's computation).
        cas_raw: Dict[str, float] = {}
        ref = self._prev_mean_flat
        ref_norm = torch.norm(ref).item()
        for hid in hids:
            f = flat[hid]
            f_norm = norms[hid]
            if ref_norm < 1e-12 or f_norm < 1e-12:
                cos = 0.0
            else:
                cos = torch.dot(f, ref).item() / (f_norm * ref_norm)
                cos = max(-1.0, min(1.0, cos))   # clamp numerical noise
            # Map from [-1, 1] → [0, 1]
            cas_raw[hid] = (cos + 1.0) / 2.0

        # ── Component 2: Loss-Convergence Score ──────────────────────────
        inv_losses = [1.0 / max(losses[hid], 1e-8) for hid in hids]
        # Stable softmax
        max_inv = max(inv_losses)
        exp_inv = [math.exp(v - max_inv) for v in inv_losses]
        total_exp = sum(exp_inv)
        lcs_raw = {hid: exp_inv[i] / total_exp for i, hid in enumerate(hids)}

        # ── Component 3: Norm Consistency Score ──────────────────────────
        norm_vals = [norms[hid] for hid in hids]
        norm_vals_sorted = sorted(norm_vals)
        median_norm = _median(norm_vals_sorted)
        abs_devs = [abs(v - median_norm) for v in norm_vals]
        mad = _median(sorted(abs_devs)) + 1e-8    # MAD + epsilon stability
        ncs_raw: Dict[str, float] = {}
        for hid in hids:
            z = abs(norms[hid] - median_norm) / mad
            ncs_raw[hid] = math.exp(-z)            # Gaussian-like penalty

        # ── Combine into per-client trust score ──────────────────────────
        trust_raw: Dict[str, float] = {}
        for hid in hids:
            t = (
                self.alpha_cas * cas_raw[hid]
                + self.alpha_lcs * lcs_raw[hid]
                + self.alpha_ncs * ncs_raw[hid]
            )
            trust_raw[hid] = max(t, self.min_trust)

        # ── Softmax over trust scores with temperature ────────────────────
        trust_vals = [trust_raw[hid] / self.trust_temperature for hid in hids]
        max_t = max(trust_vals)
        exp_t = [math.exp(v - max_t) for v in trust_vals]
        total_exp_t = sum(exp_t)
        trust_softmax = {hid: exp_t[i] / total_exp_t for i, hid in enumerate(hids)}

        # ── Modulate by sample count ──────────────────────────────────────
        total_n = sum(sample_counts.values())
        data_frac = {hid: sample_counts[hid] / max(total_n, 1) for hid in hids}
        raw_combined = {
            hid: trust_softmax[hid] * data_frac[hid] for hid in hids
        }
        total_combined = sum(raw_combined.values())
        final_weights = {
            hid: raw_combined[hid] / max(total_combined, 1e-12) for hid in hids
        }

        # ── Update rolling mean for next round's CAS ─────────────────────
        new_mean = torch.stack(list(flat.values())).mean(dim=0)
        self._prev_mean_flat = new_mean.detach().clone()

        # ── Record audit trail ────────────────────────────────────────────
        for hid in hids:
            self.states[hid].record_round(
                trust=trust_raw[hid],
                weight=final_weights[hid],
                cas=cas_raw[hid],
                lcs=lcs_raw[hid],
                ncs=ncs_raw[hid],
            )

        # ── Log ──────────────────────────────────────────────────────────
        logger.info(
            "TrustFedAvg round %d weights: %s",
            round_num + 1,
            {hid: f"{final_weights[hid]:.4f}" for hid in hids},
        )
        logger.debug(
            "  CAS: %s | LCS: %s | NCS: %s",
            {hid: f"{cas_raw[hid]:.3f}" for hid in hids},
            {hid: f"{lcs_raw[hid]:.3f}" for hid in hids},
            {hid: f"{ncs_raw[hid]:.3f}" for hid in hids},
        )

        return final_weights

    def record_loss(self, hospital_id: str, loss: float) -> None:
        """Record a hospital's training loss (for LCS in the next round)."""
        self.states[hospital_id].record_loss(loss)

    def get_all_states(self) -> Dict[str, dict]:
        """Serialise all client trust states for checkpointing / logging."""
        return {hid: s.to_dict() for hid, s in self.states.items()}

    def trust_log_dict(
        self,
        weights: Dict[str, float],
        round_num: int,
    ) -> Dict[str, float]:
        """
        Return a flat dict of tracker-ready metrics for this round's trust scores.
        """
        out: Dict[str, float] = {}
        for hid, state in self.states.items():
            if state.trust_history:
                out[f"trust/{hid}/score"] = state.trust_history[-1]
                out[f"trust/{hid}/weight"] = state.weight_history[-1]
                out[f"trust/{hid}/cas"] = state.cas_history[-1]
                out[f"trust/{hid}/lcs"] = state.lcs_history[-1]
                out[f"trust/{hid}/ncs"] = state.ncs_history[-1]
        return out

    def summary(self) -> dict:
        """High-level summary dict for the final training report."""
        return {
            "mechanism": "trust_fedavg",
            "alpha_cas": self.alpha_cas,
            "alpha_lcs": self.alpha_lcs,
            "alpha_ncs": self.alpha_ncs,
            "trust_temperature": self.trust_temperature,
            "min_trust": self.min_trust,
            "per_client": self.get_all_states(),
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _median(sorted_vals: List[float]) -> float:
    """Return the median of a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
