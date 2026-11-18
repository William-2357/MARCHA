# MARCHA: Motor-imagery Adaptation via a Robust Convex Head Adapter for EEG

Code for the Stanford CS229 project **MARCHA** by William Tan, William Wang and Eid Al Aktaa.

EEG brain–computer interfaces need a fresh block of cued calibration trials from every new
user, because EEG changes a lot from one subject or session to the next. MARCHA swaps
the usual linear probe or fine-tuned MLP for a **Convex Head Adapter (CHA)**. CHA is a
two-layer ReLU head trained through its *exact convex reformulation* with a JAX–ADMM solver,
and it sits on top of any frozen EEG encoder. **A-CHA** extends CHA by pulling the target
solve toward a variance-weighted prior pooled from the source subjects, which fixes CHA's
collapse when calibration data is scarce.

The repo also benchmarks calibration efficiency: a per-subject **k-minute sweep** on
BCIC-IV-2a under leave-one-subject-out (LOSO). It compares zero-shot baselines (LOSO, EA,
TTA) and gradient adapters (fine-tune, LoRA, EA+LoRA) with the convex heads on three
specialist backbones (EEGNet, ShallowConvNet, EEG Conformer) and three foundation backbones
(NeuroGPT, MIRepNet, LaBraM).

## Key findings

- **The CHA family has the best mean BCA on 4 of 6 backbones in the committed results
  (NeuroGPT, MIRepNet, EEGNet, ShallowConvNet).** The paper says 5 of 6. The difference is
  LaBraM, where the paper's Table 1 swaps two cells (see
  [Differences from the submitted paper](#differences-from-the-submitted-paper)). On EEG
  Conformer, LoRA (0.529) and fine-tuning (0.521) both beat the best convex head.
- **The largest gain is on NeuroGPT**, a general-purpose foundation encoder that was not
  pretrained for motor imagery. There, A-CHA reaches 0.616 mean BCA against a 0.492 LOSO
  baseline (+0.124 absolute, +25% relative), and it is the best method at every k.
- **The anchor fixes CHA's low-k collapse.** Plain CHA is near chance at k ≤ 1 min (≈0.25–0.30)
  because it sees only the target trials. A-CHA avoids the collapse. At k = 1 min it is
  already at or above every zero-shot baseline on EEGNet, ShallowConvNet and LaBraM, and on
  NeuroGPT it leads at every k.
- **Fit time does not grow with k.** The ADMM solve takes about 0.3 s at every budget
  (compile excluded, RTX 5090). LoRA and fine-tuning grow linearly with k, so the convex heads
  are the fastest methods from k ≥ 5 and 5–8× faster than the gradient adapters at k = 15.
- **No hyperparameter tuning per subject or per budget.** A-CHA uses one anchor constant
  `c = 120` everywhere, and BCA barely changes over a 4× range of `c` (paper Table 6).

<p align="center">
  <img src="results/figures/neurogpt_adapter_comparison.png" width="49%" alt="BCA vs. calibration minutes on NeuroGPT">
  <img src="results/figures/mirepnet_adapter_comparison.png" width="49%" alt="BCA vs. calibration minutes on MIRepNet">
  <br><em>Paper Fig. 1: BCA vs. minutes of target calibration data on NeuroGPT (left) and MIRepNet (right).</em>
</p>

## Method in brief

**CHA.** Take a frozen encoder $`\phi`$ and calibration features $`Z \in \mathbb{R}^{n \times d}`$.
The two-layer ReLU head $`f(z) = W_2\,\mathrm{ReLU}(W_1 z)`$ is non-convex to train and
sensitive to its seed and learning rate when only a few trials are available. Following
Pilanci & Ergen (2020) and Sahiner et al. (2021), we sample $`G`$ ReLU activation patterns
$`D_g`$ and solve the equivalent convex program (paper Eq. 2):

```math
\min_{\{v_g, w_g\}_{g=1}^{G}} \; \ell\Big(\sum_{g=1}^{G} D_g Z (v_g - w_g),\, y\Big) + \beta \sum_{g=1}^{G} \big(\|v_g\|_2 + \|w_g\|_2\big)
\quad \text{s.t.} \quad v_g, w_g \in \mathcal{K}_g = \{ v : (2D_g - I) Z v \ge 0 \}
```

The convex program is solved with the batched JAX–ADMM solver from CRONOS (Feng et al., 2024),
via `jaxcld`. Its solution is globally optimal, does not depend on the seed, and needs no
learning-rate or early-stopping choices.

**A-CHA** works in two stages:

1. **Stage 1 (source prior).** Solve CHA once per source subject. Pool the solutions into a
   mean prior $`\bar v_g, \bar w_g`$ and a per-pattern cross-subject variance
   $`\widehat{\mathrm{Var}}_s(v_g^{(s)})`$.
2. **Stage 2 (anchored calibration solve).** On the target's $`n_c`$ calibration features
   $`Z_c`$, solve the same program plus a quadratic pull toward the prior (paper Eq. 3):

```math
\min_{\{v_g, w_g\}_{g=1}^{G}} \; \tfrac{1}{2}\Big\|\sum_{g=1}^{G} D_g Z_c (v_g - w_g) - y_c\Big\|_2^2
+ \beta \sum_{g=1}^{G} \big(\|v_g\|_2 + \|w_g\|_2\big)
+ \tfrac{1}{2} \sum_{g=1}^{G} \alpha_g \big(\|v_g - \bar v_g\|_2^2 + \|w_g - \bar w_g\|_2^2\big)
\quad \text{s.t.} \quad v_g, w_g \in \mathcal{K}_g
```

with per-pattern anchor strength (paper Eq. 4)

```math
\alpha_g = \frac{c}{n_c}\,\tilde a_g, \qquad \tilde a_g \propto \frac{1}{\widehat{\mathrm{Var}}_s(v_g^{(s)}) + \varepsilon}, \qquad \frac{1}{G}\sum_{g=1}^{G} \tilde a_g = 1 .
```

The $`1/n_c`$ factor makes the anchor strong when calibration data is scarce and weaker as more
arrives. The per-pattern weight anchors directions that agree across source subjects firmly
and leaves directions where they disagree free to fit the target. The program stays convex,
so the same ADMM solver applies. The constant `c = 120` is fixed for every backbone, subject
and budget.

## Reproducing the paper

| Paper artifact | How to regenerate | Output |
|---|---|---|
| Table 1 (mean BCA) | `python scripts/sync_results_table.py` (reads committed `results/`) | the [Results](#results) block below |
| Fig. 1 / Fig. 4 (BCA vs. k) | `python scripts/plot_adapter_comparison.py --kadapt-root results/kadaptive_fixed` | `results/figures/{backbone}_adapter_comparison.png` |
| Fig. 2 (mean train-fit time vs. k) | `python scripts/aggregate_fittime.py --metric train_fit_time_warm && python scripts/plot_fittime_final.py --metric train_fit_time_warm` | `results/figures/train_fit_time_vs_k.png` |
| Fig. 5 (train-fit time per backbone) | same as Fig. 2 | `results/figures/train_fit_time_vs_k_by_model.png` |
| Fig. 3 (per-subject heatmap, NeuroGPT) | `python scripts/plot_per_subject.py --backbone neurogpt` | `results/figures/neurogpt_per_subject_heatmap.png` |
| Table 6 (anchor constant `c`) | rerun A-CHA with `KADAPT_A_BASE` ∈ {0.5, 1, 2, 4} (`c = 60 × KADAPT_A_BASE`) | new `modal_summary.json` per run |
| New accuracy runs | `scripts/run_all_modal.sh`, `scripts/run_kadaptive_modal.sh` (Modal) or `run_experiment.py` (local) | `results/…/modal_summary.json` |
| New fit-time runs | `scripts/run_fittime_modal.sh` (Modal) or `run_fittime.py` (local) | `results/fittime/…` |

Fit times in the paper are **compile-excluded, on-target training time** (`train_fit_time_warm`).
They count only the ADMM solve or the fine-tune epoch loop. Feature extraction, backbone
loading, the cached source stage and the first JAX/XLA compile are left out. See
`scripts/fittime_metrics.py`.

## Paper ↔ Code Terminology

The paper and the code name the same things differently:
the paper's names are expository, while the code's names are the **runtime API** — module
paths, method-registry strings, config keys and env vars — and are deliberately left
unchanged. Use this table to map one onto the other.

| Paper name | Code identifier(s) | What it is |
|---|---|---|
| **CHA** (Convex Head Adapter) | `cld`, `foundation_cld`, `foundation_sft_cld` (+ the `ea_*` forms); `adaptation/cld.py` | Two-layer ReLU head trained through its exact convex reformulation, solved by JAX–ADMM on frozen-encoder features |
| **A-CHA** (Anchored CHA) | `kadaptive_anchored_cld`, `foundation_sft_kadaptive_anchored_cld` (+ the `ea_*` forms); `adaptation/kadaptive_anchored_cld.py`, `adaptation/foundation_sft_kadaptive_anchored_cld.py` | CHA plus the variance-weighted quadratic anchor toward a pooled per-source-subject prior |
| **EA+CHA** / **EA+A-CHA** | any `ea_`-prefixed form of the above | Euclidean Alignment composed with the convex head |
| *(no paper name)* | `anchored_cld`, `foundation_sft_anchored_cld` (+ the `ea_*` forms) | The **earlier** warm-start anchor on source∪weighted-calibration. Superseded by A-CHA; kept for comparison and **not** reported in the paper |
| anchor strength $`c`$ | `anchor_a_base` × `anchor_n_ref`, env `KADAPT_A_BASE` | Paper Eq. (4): $`\alpha_g = \frac{c}{n_c}\tilde a_g`$ |
| per-pattern weight $`\tilde a_g`$ | `a_pattern`, selected by `anchor_mode="adaptive"` | $`\tilde a_g \propto 1/\widehat{\mathrm{Var}}_s(v_g^{(s)})`$, mean-normalized to 1 |
| Stage 1 / Stage 2 | `fit_stage1_source` / `_fit_stage2` | Source prior solve / anchored calibration solve |
| calibration budget $`k`$ (minutes) | `k_minutes`, `K`, `--k-minutes` | Minutes of labeled target calibration |
| BCA | `bca` in the result JSON | Balanced class accuracy (primary metric) |
| $`G`$ (sampled ReLU patterns) | `n_neurons` / `P_S` | Number of sampled activation patterns in the convex program |

> **On the acronym `CLD`.** The code's `cld` name is inherited from the convex-head
> implementation this project reuses via the `jaxcld` library (Feng, Tan & Pilanci, 2026 —
> "Convex Language Detection"; see the `adaptation/cld.py` docstring). It does **not** stand
> for "convex label denoising". Everywhere `cld` appears in a method name, the paper says
> **CHA**.

## Overview

The experiment design uses leave-one-subject-out (LOSO) style transfer:
- **Source data**: pooled training trials from all other subjects
- **Target calibration pool**: first 80% of the held-out subject's session-1 trials
- **Target test set**: last 20% of that same session-1 split

Methods are evaluated at K=0 (zero/few-shot) and across K in minutes of labeled calibration data.

Every run uses seed 42 with 5 repeats per K, each repeat drawing a different calibration
subset. The primary metric is balanced class accuracy (BCA), averaged over the 9 held-out
subjects. Cohen's κ is reported as well. `evaluation/metrics.py` also computes 95%
block-bootstrap confidence intervals (1000 resamples, block size 30) and a label-permutation
test (200 permutations).

## Dataset

**BCIC-IV-2a** — 9 subjects, 22 EEG channels, 4 motor imagery classes, 250 Hz, session 1 only (288 trials/subject, 72 per class).

Trials are extracted as **4-second epochs** (`epoch_len_sec=4.0`). The K-minute sweep
(`evaluation/protocols.py:minutes_to_trials`) converts minutes to a balanced trial count as
`n_trials = 15 × K` (15 four-second epochs per minute), sampled stratified across the 4 classes.

| k (min) | 0.5 | 1 | 2 | 5 | 10 | 15 | 30 |
|---|---|---|---|---|---|---|---|
| Calibration trials | 4 | 12 | 28 | 72 | 148 | 224 | 230 |
| % of pool | 2 | 5 | 12 | 31 | 64 | 97 | 100 |

The test set is always the last 20% of the held-out subject's session-1 trials (58 trials).
The calibration pool is the first 80% (230 trials). Only the labelled "T" session is used.

> **Calibration ceiling.** The target calibration pool is the first 80% of the session ≈ **230
> trials ≈ 15.3 min** of 4-second epochs. Balanced sampling saturates there, so **K=15 already
> consumes ~97% of the pool and any K ≳ 15.3 (e.g. K=30) reuses the same ~230 trials** — it is
> *not* a true 30-minute condition. Treat K > 15 as the "full calibration pool" plateau for both
> accuracy and `fit_time`.

Expected raw layout:
```
data/raw/bciciv2a/A01T.npz
data/raw/bciciv2a/A02T.npz
...
```

## Backbone Families

The paper reports six backbones (paper Table 4). Parameter counts are for the architecture
exactly as used here: 22 channels and 4-second epochs.

| Backbone | Family | Params | Features / output | Pretraining |
|---|---|---|---|---|
| EEGNet (`eegnet`) | Specialist | 3.1k | logits (4) | — (trained from scratch on source) |
| ShallowConvNet (`shallowconv`) | Specialist | 43.8k | logits (4) | — |
| EEG Conformer (`conformer`) | Specialist | 646k | logits (4) | — |
| NeuroGPT (`neurogpt`) | Foundation | 1.1M | 1024 | TUH EEG (TUEG) |
| MIRepNet (`mirepnet`) | Foundation | 5.2M | 256 | MI (BCIC-IV, OpenBMI) |
| LaBraM-base (`labram`) | Foundation | 5.8M | 200 | TUH + BCIC |

NeuroGPT's parameter count covers only the encoder and linear embedder, because the
pretrained GPT decoder is discarded during feature extraction.

### 1. Specialist Backbones (`models/specialists.py`)
`eegnet`, `shallowconv`, `conformer` (plus `deep4net`, which is implemented but not in the
paper). They output logits directly and are trained and adapted with ordinary supervised
learning.

### 2. Foundation Backbones (`models/foundations.py`)
All expose `get_features(X)` and `feature_dim` via the `FoundationBackbone` interface.

| Model | Feature dim | Key Details |
|-------|-------------|------------|
| `MIRepNet` | 256 | Maps to 45-channel template; receives native 250 Hz data (no resampling round-trip) |
| `NeuroGPT` | 1024 | Encoder + embedder portions, 22 channels, 250 Hz / 2-second chunks |
| `LaBraM` | 200 | Channel-index mapping, temporal patching, own feature normalization |

> **Sampling rates:** `MIRepNet` and `NeuroGPT` now run on native **250 Hz** data — the dataset is delivered at the backbone's native rate rather than resampled down and back up, eliminating a lossy round-trip. `LaBraM` runs at 200 Hz. Per-backbone target rates live in `BACKBONE_TARGET_SFREQ` (`data/preprocessing.py`); **delete stale caches after changing a rate.**

## Preprocessing

Per-backbone preprocessing configs (`data/preprocessing.py`):

| Backbone | Bandpass | Z-score |
|----------|----------|---------|
| `labram` | 0.1–75 Hz | No |
| `neurogpt` | 0.5–40 Hz | No |
| `mirepnet` | 4–40 Hz | No |
| Specialists | 4–40 Hz | Yes |

Default pipeline: common average reference → bandpass → notch (50 and 60 Hz) → resample → epoch extraction → per-trial per-channel z-score.

The paper's Methods section describes the specialist pipeline (CAR, 4–40 Hz, notch, z-score).
The foundation backbones use the per-backbone bandpass and sampling rate in the table above,
so their inputs match what each model was pretrained on.

Preprocessed data is cached to `data/*_cache/` locally or `/data/bciciv2a_cache` on Modal.

## Adaptation Methods

Three-tier architecture:

**Tier 1 — Specialist** adapters (backbone outputs logits directly):

| Method | Paper | File | Description |
|--------|-------|------|-------------|
| `loso` | LOSO | `adaptation/loso.py` | Zero-shot source-trained baseline |
| `ea` | EA | `adaptation/ea.py` | Euclidean Alignment baseline |
| `tta` | TTA | `adaptation/tta.py` | TENT (if BatchNorm) or T3A |
| `finetune` | Fine-tune | `adaptation/finetune.py` | Full supervised fine-tuning |
| `lora` | LoRA | `adaptation/lora.py` | LoRA adaptation |
| `ea_lora` | EA+LoRA | `adaptation/ea_lora.py` | EA + LoRA |
| `cld` | **CHA** | `adaptation/cld.py` | Convex Head Adapter — convex two-layer ReLU head via JAX–ADMM |
| `ea_cld` | **EA+CHA** | `adaptation/stacked.py` | EA + CHA |
| `anchored_cld` | — | `adaptation/anchored_cld.py` | Warm source-anchored 2-stage convex head (low-K fix; specialist analogue of `foundation_sft_anchored_cld`). Superseded by A-CHA |
| `ea_anchored_cld` | — | `adaptation/anchored_cld.py` | EA + warm source-anchored 2-stage convex head |
| `kadaptive_anchored_cld` | **A-CHA** | `adaptation/kadaptive_anchored_cld.py` | Anchored CHA — variance-weighted explicit anchor (specialist analogue of `foundation_sft_kadaptive_anchored_cld`) |
| `ea_kadaptive_anchored_cld` | **EA+A-CHA** | `adaptation/kadaptive_anchored_cld.py` | EA + Anchored CHA |

**Tier 2 — Foundation (frozen backbone)** adapters:

| Method | Paper | Description |
|--------|-------|-------------|
| `linear_probe` | linear probe | Canonical linear probe on frozen features |
| `foundation_loso` | LOSO | Linear-probe K=0 baseline |
| `foundation_ea` | EA | EA before feature extraction, then linear probe |
| `foundation_tta` | TTA | T3A on frozen backbone |
| `foundation_finetune` | Fine-tune | K=0 probe; K>0 full fine-tune |
| `foundation_lora` | LoRA | K=0 probe; K>0 LoRA on eligible layers |
| `foundation_ea_lora` | EA+LoRA | EA + LoRA |
| `foundation_cld` | **CHA** | Frozen features + convex head |
| `foundation_ea_cld` | **EA+CHA** | EA + frozen features + convex head |

**Tier 3 — Foundation SFT** adapters (source fine-tuned then frozen):

Stage 1: Fine-tune full backbone + head on pooled source data.
Stage 2: Freeze backbone; apply lightweight target adaptation.

This is the tier the paper reports for the three foundation backbones.

| Method | Paper | Description |
|--------|-------|-------------|
| `foundation_sft_loso` | LOSO | SFT backbone, no target adaptation |
| `foundation_sft_ea` | EA | EA align + SFT backbone (zero-shot) |
| `foundation_sft_tta` | TTA | SFT backbone + T3A |
| `foundation_sft_finetune` | Fine-tune | SFT backbone + full target fine-tune |
| `foundation_sft_lora` | LoRA | SFT backbone + LoRA |
| `foundation_sft_ea_lora` | EA+LoRA | EA + SFT backbone + LoRA |
| `foundation_sft_cld` | **CHA** | SFT backbone + convex head (Stage 2 on **target calibration only**) |
| `foundation_sft_ea_cld` | **EA+CHA** | EA + SFT backbone + convex head (target-only) |
| `foundation_sft_anchored_cld` | — | SFT backbone + **warm source-anchored** 2-stage convex head (low-K fix). Superseded by A-CHA; not reported in the paper |
| `foundation_sft_ea_anchored_cld` | — | EA + SFT backbone + warm source-anchored 2-stage convex head |
| `foundation_sft_kadaptive_anchored_cld` | **A-CHA** | SFT backbone + variance-weighted explicit anchor (data-relative source prior, cal-only Stage 2) |
| `foundation_sft_ea_kadaptive_anchored_cld` | **EA+A-CHA** | EA + SFT backbone + variance-weighted explicit anchor |

SFT hyperparameter defaults (hardcoded in each adapter file):
- `lr_src=1e-3`, `weight_decay=1e-4`, `max_epochs_src=200`, `patience_src=25`

### Hyperparameters (paper Table 5)

| Component | Setting | Value |
|---|---|---|
| Stage 1: source training / SFT (all backbones) | optimizer, lr, weight decay | Adam(W), 1e-3, 1e-4 |
| | max epochs / patience, val. split | 200 / 25, 0.1 |
| Fine-tune (foundation) | lr (probe / full) | 1e-3 / 1e-5, ≤100 epochs, patience 15 |
| Fine-tune (specialist) | lr | 1e-4, ≤100 epochs, patience 15 |
| LoRA (foundation) | r, α, dropout; lr | 8, 16, 0.1; 1e-3 |
| LoRA (specialist) | r, α, dropout | rank chosen by source CV, α = 2r, 0.1 |
| TTA (specialist, BatchNorm) | TENT | 10 steps, lr 1e-3, BN affine |
| TTA (foundation, LayerNorm) | T3A | confident-pseudolabel prototypes |
| CHA head | ADMM | `n_neurons=32` (4-class), rank 20, β=1e-3, ρ=0.01; 50 ADMM / 10 PCG iterations |
| A-CHA anchor | | β=1e-4, `anchor_a_base=2.0`, `anchor_n_ref=60` (c = 120) |

These are the values the paper reports. The source of truth is the constructor defaults in
each `adaptation/*.py` file. For example, the specialist source-training loop in
`adaptation/finetune.py` / `adaptation/lora.py` defaults to batch size 64 and patience 20.

### Warm source-anchored convex head (low-K fix) — *superseded by A-CHA*

Plain `foundation_sft_cld` (**CHA**) fits the convex head on the **target calibration trials
only**, which underperforms at small K (few-shot) — the same low-`k` collapse the paper
reports for CHA. The `*_anchored_cld` variants were the first fix, from
`reve_kmin_convexnn_v3.ipynb`: a **two-stage** convex solve. They are **not** the paper's
A-CHA (see the next section) and are not reported in it; they are kept for comparison.

- **Stage 1** — cold ADMM on the source pool; saves the convex-program primal `(u, v)`.
- **Stage 2** (K > 0) — rebuild `CVX_ReLU_MLP` on **source + weighted calibration**
  (calibration rows repeated to approximate a `target_mass` share of the loss), warm-started
  from the Stage-1 primal (same seed → same random hyperplanes), with the ADMM dual `lam`
  reset. The **Stage-1 feature scaler is reused** so the warm-started weights stay valid.
- K = 0 uses the Stage-1 model directly.

**Per-backbone HP tuning.** The two knobs that define Stage 2 (`beta`, `target_mass`)
are grid-searched (`{3e-4,1e-3,3e-3} × {0.15,0.35,0.55}`) on a **leak-free, source-internal
validation split** (a held-out source slice acts as a pseudo-target). HPs are **tuned once
per backbone** (not per fold): a single "primer" job per anchored variant selects and
persists them to the volume (`{backbone}_anchored_{ea_}hp_seed{seed}.json`), and the
remaining folds reuse them. EA and non-EA variants tune separately (EA whitens the feature
space). HPs are tuned on the foundation-feature space, not copied from the REVE notebook.
(This grid applies only to the *warm-anchored* variants above; the K-adaptive variants
below use a fixed `beta` and no HP grid.)

### K-adaptive source-anchored convex head — **A-CHA** in the paper

This is the paper's **A-CHA** (Anchored Convex Head Adapter); its EA composition is
**EA+A-CHA**. The warm-anchored variants above tie Stage 2 to the source *implicitly* (warm-start + a
short ADMM on source∪weighted-calibration), governed by the fixed `beta`/`target_mass`
grid. A single fixed anchor strength is brittle — too strong and it flattens at high K,
too weak and it craters at low K. The `*_kadaptive_anchored_cld` variants
(`adaptation/foundation_sft_kadaptive_anchored_cld.py` for foundation SFT,
`adaptation/kadaptive_anchored_cld.py` for specialists) replace it with an **explicit,
data-relative quadratic anchor** on the **calibration-only** solve — the paper's Eq. (3):

```math
\text{Stage 2:}\quad \min_{v}\; \tfrac{1}{2}\|F(v) - y_{\mathrm{cal}}\|_2^2 + \tfrac{a_{\mathrm{eff}}}{2}\|v - v_{\mathrm{anchor}}\|_2^2 + \beta\,\mathrm{GroupLasso}(v),
\qquad a_{\mathrm{eff}} = \frac{a_{\mathrm{base}}\, n_{\mathrm{ref}}}{n_{\mathrm{cal}}}
```

In the paper's notation this is $`\alpha_g = \frac{c}{n_c}\tilde a_g`$ (Eq. 4), with
$`c =`$ `anchor_a_base` × `anchor_n_ref`, $`n_c =`$ `n_cal`, and $`\tilde a_g`$ the
per-pattern weight below. The paper's fixed $`c = 120`$ is the `2.0 × 60.0` default.

The anchor strength `a_eff` **scales inversely with the number of calibration trials**:
strong when calibration is scarce (low K — it fills the underdetermined null space) and
receding as K grows (high K — the data dominates). Source therefore enters via the anchor
term, not by pooling rows. Defaults `anchor_a_base=2.0`, `anchor_n_ref=60.0` (product 120,
the tested-best value); `beta=1e-4` fixed, `hp_select=False`.

- **`anchor_mode="adaptive"` (default)** — the paper's variance-weighted anchor: a
  **per-pattern** prior $`a_i \propto 1/\widehat{\mathrm{Var}}_s(v_i^{(s)})`$ (paper: $`\tilde a_g`$)
  built from a multi-task per-source-subject convex solve (the paper's Stage 1): hyperplanes
  conserved across source subjects are anchored hard, variable ones are left free to fit the
  target. Needs
  `source_per_subject` (supplied via `source_cache`); falls back to the pooled-source
  **isotropic** anchor (`stage1_model.v`, strength 1) when per-subject source is unavailable.
- **`stage2_data="cal"` (default, tested best)** — Stage 2 sees calibration rows only.
  `"source_cal"` instead pools source + `target_mass`-weighted calibration (like the
  warm-anchored variant) while still applying the explicit anchor.
- **EA variants** align the source (per-subject when available), target-unlabeled, and
  calibration before the convex pipeline; the per-pattern anchor solves on the *aligned*
  per-subject source so it lives in the same whitened space.
- **K = 0** uses the Stage-1 source model directly (as with the warm-anchored variants).

When to use which: the numbers in this paragraph come from development runs, not from the
paper's tables. The K-adaptive anchor wins where source is less task-aligned (tested
NeuroGPT full-dim, cal-only, adaptive → **0.678 BCA**, beating the source∪cal union at
0.672/0.676 and `foundation_sft_lora` at 0.656; the fixed-`a` anchor scored 0.654). On a
MI-pretrained backbone whose source already spans the task (e.g. MIRepNet) the warm
source∪cal union still wins — it is **backbone-dependent**.

**A/B knobs (foundation variant).** The foundation adapter reads four env vars (baked into
the Modal image at build from the local env; defaults = tested config): `KADAPT_A_BASE`
(anchor base strength), `KADAPT_ANCHOR_MODE` (`adaptive`|`isotropic`), `KADAPT_STAGE2`
(`cal`|`source_cal`), and `KADAPT_SFT_EPOCHS` (`0` skips source fine-tuning and anchors the
**raw frozen** backbone features — SFT can overfit source and hurt cal-only transfer). The
specialist variant takes the same knobs as constructor args.

### `fit_time` semantics (foundation SFT methods)

For the SFT adapters, `fit_time` measures **only the per-K target adaptation** — the timer
starts after the frozen, source-fine-tuned backbone is ready. Source fine-tuning (cached to
disk), backbone loading, and source feature extraction are excluded so methods are comparable.
The frozen backbone, source features, and Stage-1 solve are cached across K within a job.
(Note: results predating this change bundled SFT training/model-reload into `fit_time` and are
**not** directly comparable across methods — e.g. LoRA's K=0.5 included a full source fine-tune.)

### Convex-head (CHA / A-CHA) implementation notes

The convex-head methods (`cld`, `ea_cld`, `foundation_cld`, `foundation_sft_cld`, and the
anchored variants — **CHA** and **A-CHA** in the paper) use a **JAX-based** convex ADMM head
via the `jaxcld` library.
A few non-obvious details:

- **XLA recompilation fixed via fixed-bucket padding.** The jitted PCG/ADMM solver
  recompiled on every call because the sample count `N` varied across K values
  (and across source vs. target stages), and `jit` retraces on shape changes.
  Features are now zero-padded up to a fixed bucket size (`pad_features_to_bucket`
  in `adaptation/cld.py`) so the solver compiles once and is reused. The padding is
  solution-invariant.
- **Anchored Stage 2 pads to a `target_mass`-invariant size.** In the source-anchored
  variants, `X_aug = source + repeated-calibration` changes row count with both K and
  `target_mass`, so next-multiple padding still retraced per call (a NeuroGPT EA run
  took ~7 min vs ~2 min). `fit_stage2_anchored` now pads to `n_src·(1+odds)` rounded up
  plus one bucket — a function of `n_src` and `target_mass` only (both constant for a
  sweep) — so Stage 2 compiles once. The EA variant also caches the aligned source.
- **GPU backend forced and verified.** `modal_runner.py` sets `JAX_PLATFORMS=cuda,cpu`
  on the image (cuda default; cpu kept for the Nyström pin) and the worker raises if
  `jax.default_backend()` isn't a GPU — a silent CPU fallback would run the solver
  ~10–20× slower and otherwise look identical.
- **Nyström preconditioner pinned to CPU.** On Modal GPUs, `jaxcld`'s Nyström
  preconditioner (`rand_nys_appx`) crashed with `cuSolver INTERNAL` because its
  `qr`/`cholesky`/`solve_triangular`/`svd` are cuSolver routines. These act on
  tiny (PCA-reduced) matrices, so `adaptation/_jaxcld_cpu_linalg.py` monkey-patches
  them onto the CPU while the expensive sketch matvecs and PCG/ADMM matmuls (cuBLAS)
  stay on GPU. Numerically identical to upstream.
- **PCA reduction is available but disabled by default** (`max_feat_dim=None`).
  When set, high-dim backbones (e.g. NeuroGPT's 1024-dim features) are PCA-reduced
  before the convex head to keep the ADMM weight tensors small enough for XLA. With it
  off, the convex head fits on the full feature space; only NeuroGPT (1024 > 256) was
  ever affected — all other backbones (≤256-dim) are bit-identical either way.

## Running Experiments

### Local

```bash
python run_experiment.py --dataset bciciv2a --backbone labram \
    --methods foundation_sft_lora foundation_sft_cld \
    --checkpoint-path /path/to/LaBraM.pth
```

### Modal (cloud, GPU)

```bash
modal run modal_runner.py
```

Default Modal config (set at the top of `modal_runner.py`): **LaBraM** backbone with the
default foundation-SFT method list (the baselines plus the warm `*_anchored_cld` variants),
A10G GPU, 20 concurrent jobs (`MAX_CONCURRENCY`), 5 repeats per K (`N_REPEATS`), K-minutes
sweep `[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]`. Change `BACKBONE` / `METHODS` /
`CHECKPOINT_PATH` there or via `--backbone` / `--methods` / `--checkpoint-path`. The
`*_kadaptive_anchored_cld` variants are registered but not in the default list — add them
via `--methods`; the `KADAPT_*` env vars (see above) are baked into the image at build time.

> Per the **calibration ceiling** above, the K=30 sweep point is redundant with K≈15.3 (same
> ~230 trials); it is kept only for backward-comparability with earlier runs.

The orchestrator is restart-friendly — it resumes from an existing `modal_summary.json` and checkpoints after each finished job.

## Results

<!-- BEGIN GENERATED: results-table (scripts/sync_results_table.py) -->

Mean balanced class accuracy (BCA) on BCIC-IV-2a, generated from `results/` by
`scripts/sync_results_table.py` — do not edit by hand. Zero-shot rows report k=0;
supervised rows are the mean over the k-min sweep (including k=30, matching the paper's Table 1).
**Bold** = best per backbone.

| Method | NeuroGPT | MIRepNet | LaBraM | EEGNet | ShallowNet | Conformer |
|---|---|---|---|---|---|---|
| *Zero-shot (k=0) — baselines* | | | | | | |
| LOSO | 0.492 | 0.412 | 0.341 | 0.494 | 0.435 | 0.484 |
| EA | 0.456 | 0.394 | 0.366 | 0.505 | 0.466 | 0.478 |
| TTA | 0.526 | 0.395 | 0.327 | 0.509 | 0.456 | 0.397 |
| *Supervised (k>0) — baselines* | | | | | | |
| Fine-tune | 0.511 | 0.431 | 0.343 | 0.521 | 0.486 | 0.521 |
| LoRA | 0.566 | 0.472 | 0.366 | 0.521 | 0.463 | **0.529** |
| EA+LoRA | 0.526 | 0.475 | **0.382** | 0.515 | 0.482 | 0.498 |
| *Supervised (k>0) — ours* | | | | | | |
| CHA | 0.532 | **0.492** | 0.315 | 0.466 | 0.453 | 0.408 |
| EA+CHA | 0.502 | 0.392 | 0.330 | 0.446 | 0.426 | 0.399 |
| A-CHA | **0.616** | 0.440 | 0.357 | **0.547** | 0.504 | 0.464 |
| EA+A-CHA | 0.572 | 0.459 | 0.381 | 0.538 | **0.534** | 0.453 |

Rows map onto the code as: CHA = `cld` / `foundation_sft_cld`, A-CHA =
`kadaptive_anchored_cld` / `foundation_sft_kadaptive_anchored_cld` (see
*Paper ↔ Code Terminology*). Baselines and CHA come from
`results/{foundation,specialist}/`; A-CHA from `results/kadaptive_fixed/`.

<!-- END GENERATED: results-table -->

### Keeping the table in sync

The block above is **generated**, not hand-maintained — the README's numbers used to
drift from `results/`. Regenerate it after any new run:

```bash
python scripts/sync_results_table.py              # rewrite the block
python scripts/sync_results_table.py --check      # exit 1 if stale (hook/CI entry point)
python scripts/sync_results_table.py --audit      # diff results/ against the paper's Table 1
python scripts/sync_results_table.py --exclude-k30  # average over k<=15 only
```

To make it automatic:

```bash
bash scripts/install_hooks.sh    # pre-commit hook: regenerates + blocks a stale commit
```

`.github/workflows/results-drift.yml` runs the same two checks on push. `--audit` compares
the committed results against the paper's submitted Table 1, so a rerun that moves a number
is surfaced rather than silently absorbed.

> **k=30 in the means.** The paper's Table 1 averages supervised rows over the *whole* sweep
> including k=30, and that is this script's default, so the generated block reproduces the
> submitted numbers. Per the **calibration ceiling** above, k=30 re-uses the same ~230 trials
> as k=15, so it nudges every supervised row upward; `--exclude-k30` gives the variant that
> matches the protocol described in the paper's Evaluation section.

### Differences from the submitted paper

`python scripts/sync_results_table.py --audit` currently reproduces 58 of 60 cells of the
paper's Table 1. The two that differ are in the LaBraM column, where the paper swaps two
values:

| LaBraM | Paper | `results/` |
|---|---|---|
| EA+LoRA | 0.381 | 0.382 |
| EA+A-CHA | **0.382** | 0.381 |

In the committed results, **EA+LoRA (not EA+A-CHA) is best on LaBraM**. That makes the
"CHA family leads on five of six backbones" claim **four of six** (NeuroGPT, MIRepNet,
EEGNet, ShallowConvNet), with LaBraM a near-tie that goes to EA+LoRA by 0.001. Because of
this, the `--audit` step in `.github/workflows/results-drift.yml` exits non-zero until either
the paper or `PAPER_TABLE1` in the script is corrected.

The table above also averages supervised rows over k = 30 (see the note above), while the
paper's Evaluation section describes the sweep as k ≤ 15.

## Results Layout

At runtime, `save_result` (`evaluation/results.py`) and the Modal orchestrator write per-job
JSON to `{output_dir}/{dataset}/{backbone}/{method}/`, with a per-backbone `modal_summary.json`
checkpointed after each job (on Modal, `output_dir` is `/project/results`):

```
results/                        # runtime output dir (Modal: /project/results)
  {dataset}/
    {backbone}/
      {method}/
        subject_01_k0.5.json
        ...
      summary.csv
      modal_summary.json
```

In this repo, the **curated** results are committed under `results/`, split by run rather
than by backbone family:

| Folder | Contents |
|--------|----------|
| `results/foundation/{labram,mirepnet,neurogpt}/` | Foundation-SFT baselines + **CHA**/**EA+CHA** + the superseded warm `*_anchored_cld` |
| `results/specialist/{eegnet,shallowconv,conformer}/` | Specialist baselines + **CHA**/**EA+CHA** |
| `results/kadaptive_fixed/{all six backbones}/` | **A-CHA** / **EA+A-CHA** — the run the paper reports |
| `results/kadaptive/{all six backbones}/` | An **earlier, superseded** A-CHA run (2026-06-02); kept for provenance, do not plot |
| `results/fittime/`, `results/fittime256/` | Fit-time benchmarks (3 subjects only — timing, not accuracy) |
| `results/figures/` | Generated figures |

Each backbone folder holds its `modal_summary.json` (the file consumed by the plotting
scripts under `scripts/`, e.g. `scripts/plot_results.py`). Note that the A-CHA numbers live
in a **separate** folder from the baselines: a per-backbone paper column is assembled from
`results/{foundation,specialist}/` **plus** `results/kadaptive_fixed/`.

## Key Files

| File | Role |
|------|------|
| `run_experiment.py` | Local CLI entrypoint |
| `modal_runner.py` | Modal remote orchestrator |
| `data/datasets.py` | `BCICIVDataset`, `SyntheticDataset` |
| `data/preprocessing.py` | Preprocessing pipeline + backbone configs |
| `models/specialists.py` | Specialist backbone definitions |
| `models/foundations.py` | Foundation backbone definitions |
| `adaptation/base.py` | Adapter base class |
| `adaptation/foundation_source_finetune.py` | Shared SFT building block |
| `evaluation/protocols.py` | `loso_evaluation`, `k_minute_sweep` |
| `evaluation/results.py` | JSON + CSV result writing |

## Dependencies

- PyTorch, PEFT (LoRA)
- JAX + `jaxcld` (for the CHA / A-CHA convex head)
- MNE (EEG preprocessing)
- Modal (cloud execution)

## Authors

- **William Tan** came up with the Convex Head Adapter, worked out the CHA / A-CHA
  formulation and the variance-weighted anchor, and did the literature review. The convex
  heads run on `jaxcld`, from his earlier work with Feng & Pilanci (2026).
- **William Wang** built the experimental pipeline (LoRA, LOSO, CHA, etc.), ran every
  backbone BCA and fit-time evaluation across the k-minute sweep, and made all figures and
  tables.
- **Eid Al Aktaa** designed the k-minute evaluation protocol, gathered and preprocessed the
  datasets, and ran the ablations (anchor-strength sensitivity, per-subject breakdown).

## Citation

```bibtex
@misc{tan2026marcha,
  title  = {{MARCHA}: Motor-imagery Adaptation via a Robust Convex Head Adapter for {EEG}},
  author = {Tan, William and Wang, William and Al Aktaa, Eid},
  year   = {2026},
  note   = {Stanford CS229 project}
}
```

The convex reformulation follows Pilanci & Ergen (ICML 2020), Sahiner et al. (ICLR 2021) and
Mishkin et al. (ICML 2022). The ADMM solver is CRONOS (Feng, Frangella & Pilanci, 2024).
