"""
MedTrace Experiment Tracking
=============================
Pluggable tracker abstraction with Weights & Biases, MLflow, and NoOp backends.

Architecture
------------
* ``ExperimentTracker`` — abstract interface; all public methods are safe to call
  regardless of backend state.  Failures are logged at DEBUG level and never
  propagate to the training loop.
* ``NoOpTracker``   — zero-dependency fallback; silently discards every call.
* ``MLflowTracker`` — logs to a local SQLite store (or remote URI).
* ``WandbTracker``  — logs to Weights & Biases cloud / local server.
* ``create_tracker(cfg)`` — factory; reads ``cfg.tracker.backend`` and returns the
  correct instance.  Always returns a working tracker (falls back to NoOp on error).

Metrics tracked
---------------
Per FL round (step = round number):
  round/avg_loss            Weighted average train loss across hospitals
  round/min_loss            Best (lowest) per-hospital loss
  round/max_loss            Worst per-hospital loss
  round/weight_divergence   Mean-squared deviation of hospital adapters from global
  round/aggregation_time    Seconds spent in FedAvg aggregation
  round/total_samples       Total training samples used this round
  round/time_seconds        Wall-clock time for the full round
  round/eta_minutes         Estimated time remaining (rolling average)

Per hospital, per round (step = round number):
  {hospital_id}/train_loss
  {hospital_id}/num_samples
  {hospital_id}/training_time_seconds
  {hospital_id}/lora_params_shared

Privacy budget (step = round number):
  privacy/budget_total_epsilon   The configured total epsilon budget
  privacy/budget_spent           Cumulative epsilon spent (all hospitals, max)
  privacy/budget_remaining       Epsilon still available
  privacy/budget_pct_used        Percentage of budget consumed

Evaluation (logged at step = final round):
  eval/{question_slug}/response  (MLflow: text artifact, W&B: Table)

Summary (logged once at run end):
  summary/total_training_time_minutes
  summary/final_avg_loss
  summary/final_weight_divergence
  summary/rounds_completed
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fl_config import FLConfig

logger = logging.getLogger(__name__)

# ─── Abstract Base ────────────────────────────────────────────────────────────


class ExperimentTracker(ABC):
    """
    Common interface for all experiment tracking backends.

    Callers must never inspect the concrete type — always use this interface.
    Every method is exception-safe: failures are logged at DEBUG and discarded.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def start_run(self, cfg: "FLConfig") -> None:
        """Start the experiment run and log all hyperparameters."""
        self._safe(self._start_run, cfg)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics at the given FL round."""
        self._safe(self._log, metrics, step)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log final / run-level summary metrics (shown prominently in UI)."""
        self._safe(self._log_summary, metrics)

    def log_text(self, key: str, text: str, step: int) -> None:
        """Log free-form text (model answers, notes)."""
        self._safe(self._log_text, key, text, step)

    def log_table(self, key: str, data: List[Dict[str, Any]], step: int) -> None:
        """Log a list-of-dicts as a table (eval results, per-hospital breakdown)."""
        self._safe(self._log_table, key, data, step)

    def log_artifact(self, path: str) -> None:
        """Upload a local file as a tracked artifact."""
        self._safe(self._log_artifact, path)

    def end_run(self) -> None:
        """Finalise and close the run."""
        self._safe(self._end_run)

    # ── Abstract implementations (override in subclasses) ─────────────────────

    @abstractmethod
    def _start_run(self, cfg: "FLConfig") -> None: ...

    @abstractmethod
    def _log(self, metrics: Dict[str, Any], step: int) -> None: ...

    @abstractmethod
    def _log_summary(self, metrics: Dict[str, Any]) -> None: ...

    @abstractmethod
    def _log_text(self, key: str, text: str, step: int) -> None: ...

    @abstractmethod
    def _log_table(self, key: str, data: List[Dict[str, Any]], step: int) -> None: ...

    @abstractmethod
    def _log_artifact(self, path: str) -> None: ...

    @abstractmethod
    def _end_run(self) -> None: ...

    # ── Safety wrapper ────────────────────────────────────────────────────────

    def _safe(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) and swallow any exception."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("Tracker error (non-fatal) in %s: %s", fn.__name__, exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_config(cfg: "FLConfig") -> Dict[str, Any]:
        """Return a flat dict of all config fields suitable for param logging."""
        d = cfg.to_dict()
        flat: Dict[str, Any] = {}

        def _flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(v, f"{prefix}{k}/" if prefix else f"{k}/")
            elif isinstance(obj, (list, tuple)):
                flat[prefix.rstrip("/")] = str(obj)
            else:
                flat[prefix.rstrip("/")] = obj

        _flatten(d)
        return flat

    @staticmethod
    def _slug(text: str, max_len: int = 40) -> str:
        """Convert arbitrary text to a safe metric key slug."""
        s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
        return s[:max_len]


# ─── No-Op Backend ────────────────────────────────────────────────────────────


class NoOpTracker(ExperimentTracker):
    """
    Silent no-op tracker.  Used when ``backend = "none"`` or when the
    requested backend library is unavailable.  Every call is a no-op.
    """

    def _start_run(self, cfg):   pass
    def _log(self, m, step):     pass
    def _log_summary(self, m):   pass
    def _log_text(self, k, t, s): pass
    def _log_table(self, k, d, s): pass
    def _log_artifact(self, p):  pass
    def _end_run(self):          pass


# ─── MLflow Backend ───────────────────────────────────────────────────────────


class MLflowTracker(ExperimentTracker):
    """
    MLflow experiment tracker.

    Stores metrics in a local SQLite-backed store by default (``mlflow_uri``
    in ``TrackerConfig``).  Point at ``http://host:5000`` for a shared server.

    View results::

        cd <project_root>
        mlflow ui --backend-store-uri mlflow_runs

    Then open http://localhost:5000 in your browser.
    """

    def __init__(self, cfg: "FLConfig"):
        import mlflow  # noqa: PLC0415  (deferred import)
        self._mlflow = mlflow
        self._cfg = cfg
        tc = cfg.tracker

        mlflow.set_tracking_uri(tc.mlflow_uri)
        self._experiment_name = tc.project
        self._run_name = tc.run_name or f"fl-run-{time.strftime('%Y%m%d-%H%M%S')}"
        self._tags = {f"tag/{t}": "true" for t in tc.tags}
        self._run = None

        logger.info("MLflow tracker: experiment=%r uri=%r", self._experiment_name, tc.mlflow_uri)

    def _start_run(self, cfg: "FLConfig") -> None:
        self._mlflow.set_experiment(self._experiment_name)
        self._run = self._mlflow.start_run(run_name=self._run_name, tags=self._tags)

        # Log all hyperparameters as MLflow params (strings ≤ 500 chars)
        flat = self._flatten_config(cfg)
        params = {k: str(v)[:500] for k, v in flat.items()}
        # MLflow accepts max 100 params per call — batch them
        items = list(params.items())
        for i in range(0, len(items), 100):
            self._mlflow.log_params(dict(items[i:i + 100]))

    def _log(self, metrics: Dict[str, Any], step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(key, float(value), step=step)

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        # MLflow doesn't have a separate summary concept — log at a large step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(f"summary/{key}", float(value))

    def _log_text(self, key: str, text: str, step: int) -> None:
        # Write as a small text artifact, then upload and clean up
        slug = self._slug(key)
        tmp_dir = "tmp_mlflow_artifacts"
        os.makedirs(tmp_dir, exist_ok=True)
        fname = os.path.join(tmp_dir, f"{slug}_step{step:04d}.txt")
        with open(fname, "w") as fh:
            fh.write(text)
        self._mlflow.log_artifact(fname, artifact_path="text")
        os.remove(fname)

    def _log_table(self, key: str, data: List[Dict[str, Any]], step: int) -> None:
        import json as _json
        slug = self._slug(key)
        tmp_dir = "tmp_mlflow_artifacts"
        os.makedirs(tmp_dir, exist_ok=True)
        fname = os.path.join(tmp_dir, f"{slug}_step{step:04d}.json")
        with open(fname, "w") as fh:
            _json.dump(data, fh, indent=2, default=str)
        self._mlflow.log_artifact(fname, artifact_path="tables")
        os.remove(fname)

    def _log_artifact(self, path: str) -> None:
        if os.path.exists(path):
            self._mlflow.log_artifact(path)

    def _end_run(self) -> None:
        self._mlflow.end_run()
        logger.info("MLflow run complete. View with: mlflow ui --backend-store-uri %s",
                    self._cfg.tracker.mlflow_uri)


# ─── W&B Backend ──────────────────────────────────────────────────────────────


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases experiment tracker.

    Requires a free W&B account and API key.  Set the key once with::

        wandb login

    or export ``WANDB_API_KEY=<key>`` before training.

    View results at https://wandb.ai (or your local W&B server).
    """

    def __init__(self, cfg: "FLConfig"):
        import wandb  # noqa: PLC0415
        self._wandb = wandb
        self._cfg = cfg
        tc = cfg.tracker
        self._project = tc.project
        self._run_name = tc.run_name or f"fl-run-{time.strftime('%Y%m%d-%H%M%S')}"
        self._entity = tc.wandb_entity
        self._tags = list(tc.tags)
        self._run = None

        logger.info("W&B tracker: project=%r entity=%r", self._project, self._entity)

    def _start_run(self, cfg: "FLConfig") -> None:
        self._run = self._wandb.init(
            project=self._project,
            name=self._run_name,
            entity=self._entity,
            tags=self._tags,
            config=cfg.to_dict(),
            resume="allow",
        )

    def _log(self, metrics: Dict[str, Any], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(metrics)

    def _log_text(self, key: str, text: str, step: int) -> None:
        self._wandb.log({key: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def _log_table(self, key: str, data: List[Dict[str, Any]], step: int) -> None:
        if not data:
            return
        columns = list(data[0].keys())
        rows = [[str(row.get(c, "")) for c in columns] for row in data]
        table = self._wandb.Table(columns=columns, data=rows)
        self._wandb.log({key: table}, step=step)

    def _log_artifact(self, path: str) -> None:
        if not os.path.exists(path):
            return
        name = os.path.basename(path).replace(".", "_")
        art = self._wandb.Artifact(name=name, type="model")
        art.add_file(path)
        self._wandb.log_artifact(art)

    def _end_run(self) -> None:
        if self._run is not None:
            self._run.finish()
        logger.info("W&B run complete. View at: https://wandb.ai/%s/%s",
                    self._entity or "your-entity", self._project)


# ─── Factory ──────────────────────────────────────────────────────────────────


def create_tracker(cfg: "FLConfig") -> ExperimentTracker:
    """
    Build the correct tracker from ``cfg.tracker.backend``.

    Falls back to ``NoOpTracker`` if:
    * backend is ``"none"``
    * the required library is not installed
    * initialisation raises any exception

    This guarantees the training loop always receives a working tracker.

    Example usage in fl_simulate.py::

        tracker = create_tracker(cfg)
        tracker.start_run(cfg)
        ...
        tracker.log({"round/avg_loss": 1.23}, step=round_num)
        ...
        tracker.end_run()
    """
    backend = cfg.tracker.backend

    if backend == "none":
        logger.info("Experiment tracking disabled (backend='none')")
        return NoOpTracker()

    try:
        if backend == "mlflow":
            tracker = MLflowTracker(cfg)
            logger.info("MLflow tracker initialised")
            return tracker

        if backend == "wandb":
            tracker = WandbTracker(cfg)
            logger.info("W&B tracker initialised")
            return tracker

        logger.warning("Unknown tracker backend %r — falling back to NoOp", backend)
        return NoOpTracker()

    except ImportError as exc:
        pkg = "mlflow" if backend == "mlflow" else "wandb"
        logger.warning(
            "Tracker backend %r unavailable (%s). "
            "Install with: pip install %s. Falling back to NoOp.",
            backend, exc, pkg,
        )
        return NoOpTracker()
    except Exception as exc:
        logger.warning("Tracker init failed (%s) — falling back to NoOp", exc)
        return NoOpTracker()
