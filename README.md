# Agatha: Gated Hopfield Attention for Adversarially Robust Natural Language Processing

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue)](https://2026.emnlp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Andreia Podasca** and **Anup Das**

Official implementation of **Agatha**, accepted at the **EMNLP 2026 Main Conference**.

Agatha is a drop-in modification to the Transformer encoder that achieves adversarial robustness without adversarial training. It replaces softmax attention with an iterative Hopfield mechanism and introduces a gated residual connection that forces adversarial gradients through the energy landscape, where they are contracted layer by layer.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Command-line options](#command-line-options)
- [Supported tasks](#supported-tasks)
- [Model sizes](#model-sizes)
- [Outputs](#outputs)
- [Hardware](#hardware)
- [Citation](#citation)
- [License](#license)

## Overview

Agatha changes two components of a pre-norm Transformer encoder layer.

**Iterative Hopfield attention.** The query states of all heads are refined with the modern Hopfield update until their relative change falls below 1e-4 or `--max_iterations` (T_max) updates have been made:

```math
\xi^{(0)} = Q, \qquad \xi^{(t+1)} = \mathrm{softmax}\left(\frac{\beta}{\sqrt{d}}\, \xi^{(t)} K^{\top}\right) K
```

The converged state then attends over the values, softmax(β ξ\* Kᵀ / √d) V, followed by the usual output projection. β is the inverse temperature (`--beta`).

**Gated residual connections.** The identity path of each residual connection is scaled by a gate g = σ(θ) ∈ (0, 1):

```math
h \leftarrow g_{\text{hop}}\, h + \mathrm{Hopfield}(\mathrm{LN}(h)), \qquad h \leftarrow g_{\text{ffn}}\, h + \mathrm{FFN}(\mathrm{LN}(h))
```

Gates can be fixed or learned (`--gate_mode`) and applied to the Hopfield sublayer, the feed-forward sublayer, or both (`--gate_target`). As g → 1 the model reduces to the ungated Hopfield baseline in `agatha_base.py`.

Robustness is evaluated with projected gradient descent (PGD) attacks on the input embeddings under an L2 or L∞ budget ε.

## Repository structure

| File | Description |
|---|---|
| `agatha.py` | Agatha model (`GatedHopfieldModel`), PGD attack and evaluation, training loop, command-line interface |
| `agatha_base.py` | Iterative Hopfield layer and shared components, pretrained BERT weight loader, ungated baseline (`IterativeHopfieldModel`) with Gaussian-noise evaluation |
| `benchmark_dataloader.py` | Dataset loading and task metrics for GLUE, SuperGLUE, AdvGLUE, PAWS and PAWS-X |
| `noise_utils.py` | Deterministic noise injection into embeddings (used by the baseline) |
| `run_gate_ablation.sh` | Trains and evaluates Agatha on a task, or a group of tasks, over three seeds |
| `run_batch.sh` | Sweeps β, gate value and gate target for one task |
| `requirements.txt` | Python dependencies |

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/drexel-DISCO/agatha.git
cd agatha
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, you may first install the PyTorch build that matches your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)).

Datasets, the `bert-base-uncased` tokenizer and the pretrained BERT checkpoints are downloaded from the Hugging Face Hub on first use.

## Quick start

Train Agatha on SST-2 and evaluate its robustness to PGD attacks:

```bash
python agatha.py \
    --task sst2 \
    --pretrained True \
    --beta 50 \
    --max_iterations 100 \
    --gate_init 0.1 \
    --gate_target both \
    --gate_mode learned \
    --epochs 30 \
    --patience 20 \
    --seed 42 \
    --device cuda:0
```

This initialises the encoder from `google/bert_uncased_L-2_H-128_A-2`, fine-tunes it on SST-2, and evaluates the best checkpoint on the validation set under L2 PGD attacks (20 steps) with ε ∈ {0.05, 0.5, 1, 2, 3, 5}. This matches what `./run_gate_ablation.sh sst2` runs for seed 42.

> **Note:** The argparse defaults of `agatha.py` (β = 15, T_max = 50, 10 epochs, patience 5 and a fixed gate g = 1.0, i.e. effectively no gating) are meant for quick tests. The paper's primary configuration uses β = 50, T_max = 100, 30 epochs and patience 20, as in `run_gate_ablation.sh`.

To check that everything is installed correctly, a short CPU run is enough:

```bash
python agatha.py --task sst2 --device cpu --epochs 1 --max_train_samples 512 --pgd_steps 2 --eval_epsilons 0.5
```

## Reproducing the experiments

All runs fine-tune with AdamW (learning rate 2e-5, linear warmup over the first 10% of steps followed by linear decay, gradient clipping at 1.0), batch size 32 and a maximum sequence length of 128. The checkpoint with the best score on the task's primary validation metric is kept. GLUE and SuperGLUE test labels are not public, so models are evaluated on the validation split (`validation_matched` for MNLI).

### Gate ablation across seeds

```bash
./run_gate_ablation.sh <task|group> [gate] [target] [beta] [device]

# Example: g = 0.1 on both sublayers, β = 50, seeds 42, 123 and 456
./run_gate_ablation.sh sst2 0.1 both 50 cuda:0
```

The script uses `--pretrained True`, `--gate_mode learned` (the gate starts at the given value and is updated during training), T_max = 100, 30 epochs and patience 20; β defaults to 50. Each run ends with the PGD evaluation. Besides individual tasks, it accepts the groups `glue_standard` (SST-2, CoLA, MNLI, QNLI, QQP), `glue_small` (WNLI, MRPC, RTE), `glue_all`, `superglue` (BoolQ, CB, MultiRC, WSC) and `all`. Run `./run_gate_ablation.sh help` for details; the seeds are set at the top of the script.

### Sweep over β, gate value and gate target

```bash
./run_batch.sh sst2 cuda:0
```

This sweeps β ∈ {1, 5, 10, 15, 50, 100} × g ∈ {0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} × target ∈ {hopfield, both}, running `run_gate_ablation.sh` for each configuration. Gate values are clamped to [1e-4, 0.9999].

### AdvGLUE

AdvGLUE tasks are evaluation-only. Evaluate a checkpoint trained on the corresponding GLUE task with `--train False`, passing the architecture (`--hidden_size`, `--num_layers`, `--num_heads`, `--intermediate_size`) and Hopfield settings (`--beta`, `--max_iterations`) used for training. The gate values are restored from the checkpoint.

```bash
python agatha.py \
    --task adv_sst2 \
    --train False \
    --trained_model path/to/sst2_checkpoint.pt \
    --beta 50 \
    --max_iterations 100 \
    --device cuda:0
```

### Ungated baseline

`agatha_base.py` trains the iterative Hopfield model with standard residual connections and evaluates it with Gaussian noise of scale {0, 0.5, 1, 2, 5} added to the input embeddings (`--noise_type absolute` or `percentage`):

```bash
python agatha_base.py --task sst2 --train True --pretrained True --device cuda:0

# AdvGLUE evaluation, using the SST-2 checkpoint trained above
python agatha_base.py --task adv_sst2 --device cuda:0
```

## Command-line options

Main options of `agatha.py` (run `python agatha.py --help` for the full list):

| Argument | Default | Description |
|---|---|---|
| `--task` | `sst2` | Task name (see [Supported tasks](#supported-tasks)) |
| `--pretrained` | `False` | Initialise from the pretrained BERT checkpoint that matches the architecture |
| `--pretrained_checkpoint` | auto | Hugging Face checkpoint to use instead of the auto-selected one |
| `--hidden_size` | `128` | Hidden dimension |
| `--num_layers` | `2` | Number of Hopfield encoder layers |
| `--num_heads` | `2` | Number of attention heads |
| `--intermediate_size` | `512` | Feed-forward dimension |
| `--beta` | `15.0` | Inverse temperature β of the Hopfield update (paper: 50) |
| `--max_iterations` | `50` | Maximum Hopfield iterations T_max per layer (paper: 100) |
| `--gate_init` | `1.0` | Gate value g ∈ (0, 1); lower values contract gradients more |
| `--gate_mode` | `fixed` | `fixed` or `learned` gates |
| `--gate_target` | `hopfield` | Gate the `hopfield`, `ffn` or `both` residual connections |
| `--epochs` | `10` | Maximum number of training epochs |
| `--patience` | `5` | Early-stopping patience (epochs) |
| `--pgd_steps` | `20` | Number of PGD steps |
| `--pgd_alpha_factor` | `0.3` | PGD step size α as a fraction of ε |
| `--pgd_norm` | `l2` | Perturbation norm: `l2` or `linf` |
| `--eval_epsilons` | `0.05,0.5,1.0,2.0,3.0,5.0` | Perturbation budgets ε |
| `--train` | `True` | Set to `False` to only evaluate a checkpoint (see `--trained_model`) |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | `cuda`, `cuda:N` or `cpu` |

## Supported tasks

| Benchmark | Tasks |
|---|---|
| GLUE | `cola`, `sst2`, `mrpc`, `qqp`, `mnli`, `qnli`, `rte`, `wnli` |
| SuperGLUE | `boolq`, `cb`, `multirc`, `wsc` |
| AdvGLUE (evaluation only) | `adv_sst2`, `adv_qqp`, `adv_mnli`, `adv_qnli`, `adv_rte` |
| PAWS / PAWS-X | `paws`, `paws_swap`, `paws_unlabeled`, `pawsx_en`, `pawsx_de`, `pawsx_es`, `pawsx_fr`, `pawsx_zh`, `pawsx_ja`, `pawsx_ko` |

`benchmark_dataloader.py` also has configurations for further tasks (e.g. STS-B, COPA, ReCoRD, WiC, ANLI and SWAG). Run `python benchmark_dataloader.py` to list them all.

## Model sizes

The paper evaluates three scales. With `--pretrained True`, the checkpoint is selected automatically from the number of layers, heads and hidden size:

| Name | Layers | Heads | Hidden | FFN | Parameters | Pretrained checkpoint |
|---|---|---|---|---|---|---|
| TinyBERT (base) | 2 | 2 | 128 | 512 | 4.37M | `google/bert_uncased_L-2_H-128_A-2` |
| TinyBERT (scaled) | 4 | 4 | 256 | 1024 | 11.1M | `google/bert_uncased_L-4_H-256_A-4` |
| BERT-base | 12 | 12 | 768 | 3072 | 109M | `bert-base-uncased` |

`run_gate_ablation.sh` uses the scaled model for MultiRC and WSC and the base model for all other tasks. To select the scaled model manually:

```bash
python agatha.py --task multirc --hidden_size 256 --num_layers 4 --num_heads 4 --intermediate_size 1024 --pretrained True ...
```

## Outputs

`agatha.py` names each run with a tag of the form `{task}_L{layers}_H{heads}_b{beta}_g{gate_init}_iter{max_iterations}[_{checkpoint}]_seed{seed}` and writes:

| File | Contents |
|---|---|
| `<model_dir>/<tag>.pt` | Best checkpoint: model weights, validation metrics, epoch and gate values |
| `<out_dir>/training_curves_<tag>.json` | Per-epoch training loss, validation metric, gate values and Hopfield iteration / convergence statistics |
| `<out_dir>/<tag>.json` | For each ε: clean and adversarial task metrics (`clean_*`, `adv_*`), attack success rate (`asr`) and average Hopfield iterations on clean and adversarial inputs. Also the final gate values and an estimate of the spectral radius of the first gated Hopfield sublayer on clean data (`global_clean_rho`) |

`--model_dir` defaults to `./models_gated` and `--out_dir` defaults to `--model_dir`. The baseline (`agatha_base.py`) writes `agatha_base_<task>.pt` and `agatha_base_<task>.json` to `<model_dir>/L{layers}_H{heads}/`.

## Hardware

All experiments in the paper were conducted on a single NVIDIA RTX 3090 GPU (24 GB VRAM).

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{podasca2026agatha,
  title={Agatha: Gated Hopfield Attention for Adversarially Robust Natural Language Processing},
  author={Podasca, Andreia and Das, Anup},
  booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026},
  note = {available at \textcolor{blue}{https://github.com/drexel-DISCO/agatha}},
}
```

## License

This project is released under the [MIT License](LICENSE).
