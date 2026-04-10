"""
Generates MedTrace_FL_Colab.ipynb

The notebook is a pure entry point:
  - clones github.com/vivek797029/MEDTRACE
  - installs from requirements.txt
  - imports and calls run_simulation() from the real codebase
  - zero inline training logic

Run:
    python3 build_colab_notebook.py
"""
import json

REPO_URL = "https://github.com/vivek797029/MEDTRACE.git"
REPO_DIR = "/content/MEDTRACE"
SRC_DIR  = f"{REPO_DIR}/src"

cells = []


def md(*lines):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": list(lines),
    })


def code(*lines):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": list(lines),
        "execution_count": None,
        "outputs": [],
    })


# ─── Cell 1: Title ────────────────────────────────────────────────────────────
md(
    "# MedTrace FL — Federated Learning for Privacy-Preserving Medical AI\n",
    "\n",
    "**Pure entry point** — all training logic lives in the GitHub repository.\n",
    "Nothing is duplicated here. Runs end-to-end with a single `Run all`.\n",
    "\n",
    "| What runs | Where it lives |\n",
    "|---|---|\n",
    "| `run_simulation()` | `src/fl_simulate.py` |\n",
    "| `FLConfig` dataclasses | `src/fl_config.py` |\n",
    "| `AdaptiveDPMechanism` | `src/fl_adaptive_dp.py` |\n",
    "| Evaluation + plots | `src/fl_evaluator.py`, `src/fl_plots.py` |\n",
    "\n",
    "**Auto-resume:** every round is checkpointed to Google Drive.  \n",
    "If Colab disconnects, click **Runtime → Run all** — training resumes automatically.\n",
    "\n",
    "---\n",
    "## GPU unit budget (65 units)\n",
    "| GPU | Units / hr | 10 rounds (est.) | Recommendation |\n",
    "|-----|-----------|------------------|----------------|\n",
    "| T4  | ~1.8      | ~5–7 units ✅    | **Use this** |\n",
    "| L4  | ~2.8      | ~4–6 units ✅    | Also fine |\n",
    "| A100 | ~9.6     | ~9–11 units ⚠️  | Not recommended |\n",
    "\n",
    "> **Before running:** Runtime → Change runtime type → **T4 GPU**\n",
)

# ─── Cell 2: Step 1 header ────────────────────────────────────────────────────
md("## Step 1 — Verify GPU and mount Google Drive\n")

# ─── Cell 3: GPU check + Drive mount ──────────────────────────────────────────
code(
    "import os, sys, torch\n",
    "\n",
    "# ── GPU check ─────────────────────────────────────────────────────────────\n",
    "if not torch.cuda.is_available():\n",
    "    raise SystemExit(\n",
    "        'No GPU detected.\\n'\n",
    "        'Go to Runtime → Change runtime type → T4 GPU, then Run All again.'\n",
    "    )\n",
    "\n",
    "gpu_name = torch.cuda.get_device_name(0)\n",
    "gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "print(f'GPU  : {gpu_name}')\n",
    "print(f'VRAM : {gpu_mem:.1f} GB')\n",
    "\n",
    "# ── Mount Google Drive (for persistent checkpoints) ───────────────────────\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive', force_remount=True)\n",
    "print('Drive : mounted')\n",
)

# ─── Cell 4: Step 2 header ────────────────────────────────────────────────────
md("## Step 2 — Clone repository and install dependencies\n")

# ─── Cell 5: Clone + install + path setup ─────────────────────────────────────
code(
    "import subprocess, importlib\n",
    "\n",
    "REPO_URL = '" + REPO_URL + "'\n",
    "REPO_DIR = '" + REPO_DIR + "'\n",
    "SRC_DIR  = '" + SRC_DIR  + "'\n",
    "\n",
    "# ── Clone or pull ─────────────────────────────────────────────────────────\n",
    "if os.path.exists(REPO_DIR):\n",
    "    print('Repository already present — pulling latest...')\n",
    "    result = subprocess.run(\n",
    "        ['git', '-C', REPO_DIR, 'pull', '--ff-only'],\n",
    "        capture_output=True, text=True\n",
    "    )\n",
    "    print(result.stdout.strip() or 'Already up to date.')\n",
    "else:\n",
    "    print(f'Cloning {REPO_URL} ...')\n",
    "    subprocess.run(\n",
    "        ['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR],\n",
    "        check=True\n",
    "    )\n",
    "    print('Clone complete.')\n",
    "\n",
    "# ── Install from requirements.txt ─────────────────────────────────────────\n",
    "req = os.path.join(REPO_DIR, 'requirements.txt')\n",
    "print('\\nInstalling dependencies (this takes ~2 min on first run)...')\n",
    "subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '-q', '-r', req],\n",
    "    check=True\n",
    ")\n",
    "print('Install complete.')\n",
    "\n",
    "# ── Add src/ to Python path ───────────────────────────────────────────────\n",
    "if SRC_DIR not in sys.path:\n",
    "    sys.path.insert(0, SRC_DIR)\n",
    "\n",
    "# ── Verify all project modules are importable ─────────────────────────────\n",
    "print('\\nModule check:')\n",
    "for mod_name in [\n",
    "    'fl_config', 'fl_simulate', 'fl_adaptive_dp',\n",
    "    'fl_evaluator', 'fl_plots', 'fl_tracker',\n",
    "]:\n",
    "    importlib.import_module(mod_name)\n",
    "    print(f'  OK  {mod_name}')\n",
    "\n",
    "print('\\nSetup complete — ready to train.')\n",
)

# ─── Cell 6: Step 3 header ────────────────────────────────────────────────────
md(
    "## Step 3 — Configure the experiment\n",
    "\n",
    "Edit this cell to change hyperparameters.  \n",
    "Uses the real typed `FLConfig` dataclasses from the repository.\n",
    "\n",
    "**Key parameters:**\n",
    "- `fl_rounds` — number of federated rounds (10 = full run, 2 = quick test)\n",
    "- `hospitals` — number of hospital clients (3 by default)\n",
    "- `dp.epsilon` — privacy budget (lower = more private, less accurate)\n",
    "- `adaptive_dp.enabled` — novel per-client adaptive noise allocation\n",
)

# ─── Recovery guard (reused in every cell that needs project modules) ────────
RECOVERY = (
    "# ── Auto-recovery: re-clone + re-add path if kernel restarted ───────────\n"
    "import subprocess, sys, os as _os\n"
    "_REPO = '/content/MEDTRACE'\n"
    "_SRC  = f'{_REPO}/src'\n"
    "if not _os.path.exists(_REPO):\n"
    "    print('Kernel was restarted — re-cloning repository...')\n"
    "    subprocess.run(['git','clone','--depth','1',\n"
    "        'https://github.com/vivek797029/MEDTRACE.git', _REPO], check=True)\n"
    "    subprocess.run([sys.executable,'-m','pip','install','-q','-r',\n"
    "        f'{_REPO}/requirements.txt'], check=True)\n"
    "    print('Re-clone complete.')\n"
    "if _SRC not in sys.path:\n"
    "    sys.path.insert(0, _SRC)\n"
    "\n"
)

# ─── Cell 7: Configuration ────────────────────────────────────────────────────
code(
    RECOVERY,
    "import math\n",
    "from fl_config import (\n",
    "    FLConfig, DPConfig, AdaptiveDPConfig,\n",
    "    LoRAConfig, TrainingConfig, EvalConfig,\n",
    "    TrackerConfig, HospitalRegistry,\n",
    ")\n",
    "\n",
    "# ── Persistent paths (everything saved to your Google Drive) ──────────────\n",
    "DRIVE_ROOT  = '/content/drive/MyDrive/MedTrace'\n",
    "OUTPUT_DIR  = f'{DRIVE_ROOT}/outputs'\n",
    "CKPT_DIR    = f'{DRIVE_ROOT}/checkpoints'\n",
    "\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "os.makedirs(CKPT_DIR,   exist_ok=True)\n",
    "\n",
    "# ── Build experiment config ───────────────────────────────────────────────\n",
    "cfg = FLConfig(\n",
    "    # Model\n",
    "    base_model   = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',\n",
    "\n",
    "    # Federated learning\n",
    "    fl_rounds    = 10,      # set to 2 for a quick smoke-test\n",
    "    local_epochs = 1,\n",
    "    hospitals    = HospitalRegistry.build(3),   # 3 specialty hospitals\n",
    "\n",
    "    # LoRA adapter\n",
    "    lora = LoRAConfig(r=8, alpha=16, dropout=0.05),\n",
    "\n",
    "    # Differential privacy (epsilon, delta)-DP via Gaussian mechanism\n",
    "    dp = DPConfig(\n",
    "        enabled       = True,\n",
    "        epsilon       = 8.0,    # lower = more private, more noise\n",
    "        delta         = 1e-5,\n",
    "        max_grad_norm = 1.0,\n",
    "    ),\n",
    "\n",
    "    # Adaptive per-client DP (novel contribution)\n",
    "    adaptive_dp = AdaptiveDPConfig(\n",
    "        enabled              = True,\n",
    "        ema_alpha            = 0.1,   # gradient sensitivity smoothing\n",
    "        min_epsilon_fraction = 0.1,   # prevents any hospital getting 0 budget\n",
    "    ),\n",
    "\n",
    "    # Training\n",
    "    training = TrainingConfig(\n",
    "        batch_size                  = 4,\n",
    "        gradient_accumulation_steps = 4,   # effective batch size = 16\n",
    "        learning_rate               = 2e-4,\n",
    "        max_length                  = 512,\n",
    "    ),\n",
    "\n",
    "    # Evaluation: MCQ log-prob scoring every 5 rounds\n",
    "    eval = EvalConfig(\n",
    "        enabled             = True,\n",
    "        eval_every_n_rounds = 5,\n",
    "        num_eval_samples    = 200,\n",
    "    ),\n",
    "\n",
    "    # Experiment tracking: 'none' | 'mlflow' | 'wandb'\n",
    "    tracker = TrackerConfig(backend='none'),\n",
    "\n",
    "    # Output paths — all four must point to Drive so they survive disconnection\n",
    "    output_dir          = OUTPUT_DIR,\n",
    "    global_model_dir    = f'{OUTPUT_DIR}/global_model',\n",
    "    hospital_models_dir = f'{OUTPUT_DIR}/hospital_models',\n",
    "    metrics_dir         = f'{OUTPUT_DIR}/metrics',\n",
    ")\n",
    "\n",
    "# ── Print summary ─────────────────────────────────────────────────────────\n",
    "print('Experiment configuration')\n",
    "print(f'  Model       : {cfg.base_model}')\n",
    "print(f'  Hospitals   : {cfg.num_hospitals}')\n",
    "for hid, h in cfg.hospitals.items():\n",
    "    print(f'    {hid}: {h.name} ({h.location})')\n",
    "print(f'  FL rounds   : {cfg.fl_rounds}')\n",
    "print(f'  LoRA        : r={cfg.lora.r}  alpha={cfg.lora.alpha}  '\n",
    "      f'dropout={cfg.lora.dropout}')\n",
    "eff_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps\n",
    "print(f'  Batch       : {cfg.training.batch_size} x '\n",
    "      f'{cfg.training.gradient_accumulation_steps} = {eff_batch} effective')\n",
    "if cfg.dp.enabled:\n",
    "    per_round = cfg.dp.epsilon / math.sqrt(cfg.fl_rounds)\n",
    "    print(f'  DP          : enabled  epsilon={cfg.dp.epsilon}  '\n",
    "          f'delta={cfg.dp.delta}  sigma={cfg.dp.sigma:.4f}  '\n",
    "          f'C={cfg.dp.max_grad_norm}')\n",
    "    print(f'  Per-round e : {per_round:.4f}  (advanced composition)')\n",
    "    print(f'  Adaptive DP : {cfg.adaptive_dp.enabled}')\n",
    "else:\n",
    "    print('  DP          : disabled (no-privacy baseline)')\n",
    "print(f'  Output dir  : {cfg.output_dir}')\n",
    "print(f'  Checkpoints : {CKPT_DIR}')\n",
)

# ─── Cell 8: Checkpoint status check ─────────────────────────────────────────
md("## Step 4 — Check existing progress & run federated training\n")

# ─── Cell 8b: Checkpoint status ───────────────────────────────────────────────
code(
    "# ── How many rounds already done? ───────────────────────────────────────\n",
    "marker = os.path.join(CKPT_DIR, 'last_round.txt')\n",
    "if os.path.exists(marker):\n",
    "    with open(marker) as _f:\n",
    "        _last = int(_f.read().strip())\n",
    "    print(f'Checkpoint found — last completed round: {_last + 1} / {cfg.fl_rounds}')\n",
    "    if _last + 1 >= cfg.fl_rounds:\n",
    "        print('✅ ALL ROUNDS COMPLETE — training is done!')\n",
    "    else:\n",
    "        print(f'Training will RESUME from round {_last + 2}.')\n",
    "else:\n",
    "    print('No checkpoint found — training will START from round 1.')\n",
    "\n",
    "# ── List saved model rounds ───────────────────────────────────────────────\n",
    "_mdir = cfg.global_model_dir\n",
    "if os.path.exists(_mdir):\n",
    "    _rounds = sorted(d for d in os.listdir(_mdir) if d.startswith('round_'))\n",
    "    if _rounds:\n",
    "        print(f'\\nSaved model rounds ({len(_rounds)} / {cfg.fl_rounds}):')\n",
    "        for _r in _rounds:\n",
    "            _sz = sum(\n",
    "                os.path.getsize(os.path.join(_mdir, _r, _f))\n",
    "                for _f in os.listdir(os.path.join(_mdir, _r))\n",
    "            ) / (1024 * 1024)\n",
    "            print(f'  {_r} — {_sz:.1f} MB')\n",
    "    else:\n",
    "        print('No model rounds saved yet.')\n",
    "else:\n",
    "    print('No model directory yet.')\n",
    "\n",
    "# ── Epoch breakdown ───────────────────────────────────────────────────────\n",
    "print(f'\\nTraining plan:')\n",
    "print(f'  {cfg.fl_rounds} rounds × {cfg.num_hospitals} hospitals × {cfg.local_epochs} local epoch(s)')\n",
    "print(f'  Progress log format:  Round: X / {cfg.fl_rounds}')\n",
    "print(f'  Each round = {100 / cfg.fl_rounds:.0f}% of total training')\n",
)

# ─── Cell 9: Training ─────────────────────────────────────────────────────────
code(
    RECOVERY,
    "from fl_simulate import run_simulation, setup_logging\n",
    "import logging\n",
    "\n",
    "setup_logging(logging.INFO)\n",
    "\n",
    "report = run_simulation(\n",
    "    cfg,\n",
    "    checkpoint_dir = CKPT_DIR,\n",
    ")\n",
    "\n",
    "# ── Summary ───────────────────────────────────────────────────────────────\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('Training complete')\n",
    "print(f'  Time    : {report[\"total_training_time\"] / 60:.1f} min')\n",
    "print(f'  Rounds  : {len(report[\"round_metrics\"])}')\n",
    "if report['round_metrics']:\n",
    "    last = report['round_metrics'][-1]\n",
    "    print(f'  Loss    : {last[\"avg_loss\"]:.4f}')\n",
    "    print(f'  Diverge : {last[\"weight_divergence\"]:.6f}')\n",
    "if 'adaptive_dp_summary' in report:\n",
    "    print('  DP budget per hospital:')\n",
    "    for hid, s in report['adaptive_dp_summary']['per_client'].items():\n",
    "        pct = s['budget_spent'] / cfg.dp.epsilon * 100\n",
    "        print(f'    {hid}: {s[\"budget_spent\"]:.3f} / {cfg.dp.epsilon}  ({pct:.1f}%)')\n",
    "print('=' * 60)\n",
)

# ─── Cell 10: Step 5 header ───────────────────────────────────────────────────
md(
    "## Step 5 — Plot training curves\n",
    "\n",
    "Uses `EvalAccumulator` and `ResultsPlotter` from the repository.  \n",
    "Saved to `{OUTPUT_DIR}/plots/` on Google Drive.\n",
)

# ─── Cell 11: Plots ───────────────────────────────────────────────────────────
code(
    RECOVERY,
    "# ── Load report from Drive if training cell crashed this session ─────────\n",
    "import json as _json\n",
    "if 'report' not in dir() or report is None:\n",
    "    _rpath = os.path.join(cfg.metrics_dir, 'fl_training_report.json')\n",
    "    if os.path.exists(_rpath):\n",
    "        with open(_rpath) as _f:\n",
    "            report = _json.load(_f)\n",
    "        print(f'Loaded training report from Drive: {_rpath}')\n",
    "    else:\n",
    "        raise FileNotFoundError(\n",
    "            f'No training report at {_rpath}. Run the training cell first.'\n",
    "        )\n",
    "\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "from fl_evaluator import EvalAccumulator, EvalResult\n",
    "from fl_plots    import ResultsPlotter\n",
    "\n",
    "# ── Build EvalAccumulator from training report ────────────────────────────\n",
    "acc = EvalAccumulator()\n",
    "acc.set_metadata(**{k: v for k, v in report['config'].items()\n",
    "                    if isinstance(v, (str, int, float, bool))})\n",
    "\n",
    "# Per-round eval accuracy (if available from evaluation runs)\n",
    "eval_by_round = {\n",
    "    er.get('round_num', 0): er.get('accuracy', 0.0)\n",
    "    for er in report.get('eval_results', [])\n",
    "    if isinstance(er, dict)\n",
    "}\n",
    "\n",
    "for rm in report['round_metrics']:\n",
    "    rn = rm['round']\n",
    "    acc.add_eval_result(EvalResult(\n",
    "        run_label            = 'MedTrace FL',\n",
    "        round_num            = rn,\n",
    "        accuracy             = eval_by_round.get(rn, 0.0),\n",
    "        loss                 = rm['avg_loss'],\n",
    "        perplexity           = 0.0,\n",
    "        num_eval_samples     = rm['total_samples'],\n",
    "        elapsed_seconds      = rm['aggregation_time'],\n",
    "        privacy_budget_spent = 0.0,\n",
    "    ))\n",
    "    for hid, m in rm.get('hospital_metrics', {}).items():\n",
    "        acc.add_client_metrics(hid, m)\n",
    "\n",
    "# ── Render plots ──────────────────────────────────────────────────────────\n",
    "plot_dir = os.path.join(OUTPUT_DIR, 'plots')\n",
    "os.makedirs(plot_dir, exist_ok=True)\n",
    "plotter  = ResultsPlotter(output_dir=plot_dir, fmt='png', dpi=120)\n",
    "paths    = plotter.save_all(acc)\n",
    "\n",
    "for path in paths:\n",
    "    img = plt.imread(path)\n",
    "    fig, ax = plt.subplots(figsize=(11, 4))\n",
    "    ax.imshow(img)\n",
    "    ax.axis('off')\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    print(f'Saved: {path}')\n",
    "\n",
    "# ── Persist results JSON ──────────────────────────────────────────────────\n",
    "results_path = os.path.join(OUTPUT_DIR, 'results.json')\n",
    "acc.save_json(results_path)\n",
    "print(f'\\nResults JSON: {results_path}')\n",
)

# ─── Cell 12: Step 6 header ───────────────────────────────────────────────────
md(
    "## Step 6 — Evaluate the trained model\n",
    "\n",
    "Loads the final federated global model and runs inference on 3 clinical questions.  \n",
    "The model is kept in memory — **Step 7** (interactive chat) reuses it without reloading.\n",
)

# ─── Cell 13: Load model + eval questions ─────────────────────────────────────
code(
    RECOVERY,
    "# ── Reset CUDA context before loading model ──────────────────────────────\n",
    "# If the training cell crashed with a CUDA error, the GPU context is\n",
    "# corrupted. Reset it here so model loading works cleanly.\n",
    "import gc, ctypes\n",
    "gc.collect()\n",
    "try:\n",
    "    import torch as _t\n",
    "    if _t.cuda.is_available():\n",
    "        _t.cuda.synchronize()\n",
    "        _t.cuda.empty_cache()\n",
    "except Exception:\n",
    "    pass\n",
    "\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "from peft import PeftModel\n",
    "\n",
    "# ── Locate the final global model on Drive ────────────────────────────────\n",
    "_target = os.path.join(cfg.global_model_dir, f'round_{cfg.fl_rounds - 1}')\n",
    "if not os.path.exists(_target):\n",
    "    # Fall back to the highest available round\n",
    "    _available = sorted(\n",
    "        int(d.split('_')[1])\n",
    "        for d in os.listdir(cfg.global_model_dir)\n",
    "        if d.startswith('round_')\n",
    "        and os.path.isdir(os.path.join(cfg.global_model_dir, d))\n",
    "    )\n",
    "    if not _available:\n",
    "        raise FileNotFoundError(\n",
    "            f'No model rounds found in {cfg.global_model_dir}. '\n",
    "            'Did training complete?'\n",
    "        )\n",
    "    _target = os.path.join(cfg.global_model_dir, f'round_{_available[-1]}')\n",
    "    print(f'Note: round {cfg.fl_rounds - 1} not found; loading round {_available[-1]}')\n",
    "\n",
    "print(f'Loading model from: {_target}')\n",
    "_device    = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "_tokenizer = AutoTokenizer.from_pretrained(_target)\n",
    "_base      = AutoModelForCausalLM.from_pretrained(\n",
    "    cfg.base_model,\n",
    "    torch_dtype = torch.float16,   # fp16 saves ~4 GB VRAM on T4\n",
    "    device_map  = 'auto',\n",
    ")\n",
    "_model = PeftModel.from_pretrained(_base, _target)\n",
    "_model.eval()\n",
    "print('Model ready.\\n')\n",
    "\n",
    "# ── Shared inference helper (reused in Step 7) ────────────────────────────\n",
    "def ask(question: str, max_new_tokens: int = 400) -> str:\n",
    "    prompt = (\n",
    "        f'<|system|>\\n{cfg.system_msg}</s>\\n'\n",
    "        f'<|user|>\\n{question}</s>\\n'\n",
    "        f'<|assistant|>\\n'\n",
    "    )\n",
    "    inputs = _tokenizer(prompt, return_tensors='pt').to(_device)\n",
    "    with torch.no_grad():\n",
    "        out = _model.generate(\n",
    "            **inputs,\n",
    "            max_new_tokens     = max_new_tokens,\n",
    "            temperature        = 0.7,\n",
    "            do_sample          = True,\n",
    "            repetition_penalty = 1.1,\n",
    "            pad_token_id       = _tokenizer.eos_token_id,\n",
    "        )\n",
    "    return _tokenizer.decode(\n",
    "        out[0][inputs['input_ids'].shape[1]:],\n",
    "        skip_special_tokens=True,\n",
    "    )\n",
    "\n",
    "# ── Clinical evaluation questions ────────────────────────────────────────\n",
    "_eval_questions = [\n",
    "    'A 62-year-old man with hypertension has crushing chest pain radiating '\n",
    "    'to the left arm, diaphoresis, and nausea for 45 min. ECG shows ST '\n",
    "    'elevation in II, III, aVF. What is the immediate management?',\n",
    "\n",
    "    'A 55-year-old woman has sudden left-sided weakness, facial droop, and '\n",
    "    'slurred speech for 2 hours. CT head shows no hemorrhage. Next step?',\n",
    "\n",
    "    'A returned traveller from sub-Saharan Africa has cyclic fever, rigors, '\n",
    "    'and splenomegaly for 5 days. Blood smear shows ring-form trophozoites. '\n",
    "    'Treatment?',\n",
    "]\n",
    "\n",
    "for i, q in enumerate(_eval_questions, 1):\n",
    "    print(f'Q{i}: {q}')\n",
    "    print(f'A:  {ask(q)[:600]}')\n",
    "    print('─' * 70)\n",
)

# ─── Cell 14: Step 7 header ───────────────────────────────────────────────────
md(
    "## Step 7 — Interactive chat (optional)\n",
    "\n",
    "Reuses the model already loaded in Step 6 (no extra VRAM cost).  \n",
    "Type any clinical question, or type `quit` to exit.\n",
)

# ─── Cell 15: Interactive chat ────────────────────────────────────────────────
code(
    "# Reuses _model, _tokenizer, _device, and ask() from Step 6\n",
    "print('MedTrace Clinical Reasoning — Interactive Chat')\n",
    "print('Enter a clinical question, or type \"quit\" to exit.\\n')\n",
    "\n",
    "while True:\n",
    "    try:\n",
    "        question = input('Question: ').strip()\n",
    "    except (EOFError, KeyboardInterrupt):\n",
    "        break\n",
    "    if not question or question.lower() in ('quit', 'exit', 'q'):\n",
    "        break\n",
    "    print(f'\\n{ask(question, max_new_tokens=500)}')\n",
    "    print('─' * 70)\n",
    "\n",
    "# ── Free GPU memory ───────────────────────────────────────────────────────\n",
    "del _model, _base, _tokenizer\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
    "print('Memory freed.')\n",
)

# ─── Write notebook ───────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "MedTrace_FL_Colab.ipynb",
        },
    },
    "cells": cells,
}

out_path = "MedTrace_FL_Colab.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Written : {out_path}")
print(f"Cells   : {len(cells)}")
