#!/usr/bin/env python3
"""
Agatha: Gated Hopfield Attention for adversarially robust NLP.

Defines ``GatedHopfieldModel`` (Agatha), trains it on a GLUE / SuperGLUE /
PAWS task and evaluates its robustness to PGD attacks in the input embedding
space.

Example (the configuration used by run_gate_ablation.sh):

    python agatha.py --task sst2 --pretrained True --beta 50 \
        --max_iterations 100 --gate_init 0.1 --gate_target both \
        --gate_mode learned --epochs 30 --patience 20 --device cuda:0
"""

import argparse
import json
import math
import os

# Allow slow connections when downloading checkpoints and datasets from the
# Hugging Face Hub. This must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "200")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from agatha_base import (
    DEFAULT_MODEL_CONFIG,
    FeedForward,
    HopfieldPooling,
    TrueHopfieldLayer,
    get_pretrained_checkpoint,
    load_pretrained_bert_weights,
    set_seed,
)
from benchmark_dataloader import TASK_CONFIGS, compute_metrics, load_data


# ============================================================================
# ResidualGate
# ============================================================================

class ResidualGate(nn.Module):
    """
    Gated residual connection: ``g * residual + sublayer_out``.

    The gate g = sigmoid(gate_param) lies in (0, 1). It is stored as a logit
    initialised from ``gate_init`` (clamped to [1e-4, 0.9999]). In ``fixed``
    mode the logit is a buffer; in ``learned`` mode it is a trainable
    parameter. A gate close to 1 recovers the standard residual connection.
    """

    def __init__(self, gate_init: float = 1.0, gate_mode: str = 'fixed'):
        super().__init__()
        assert gate_mode in ('fixed', 'learned'), \
            f"gate_mode must be 'fixed' or 'learned', got '{gate_mode}'"
        self.gate_mode = gate_mode

        g = max(min(gate_init, 0.9999), 0.0001)
        init_val = math.log(g / (1.0 - g))
        param = torch.tensor(float(init_val))

        if gate_mode == 'learned':
            self.gate_param = nn.Parameter(param)
        else:
            self.register_buffer('gate_param', param)

    @property
    def gate_value(self) -> float:
        return torch.sigmoid(self.gate_param).item()

    def forward(self, residual: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_param)
        return gate * residual + sublayer_out


# ============================================================================
# GatedHopfieldModel
# ============================================================================

class GatedHopfieldModel(nn.Module):
    """
    Agatha encoder for sequence classification.

    A pre-norm encoder whose self-attention is replaced by iterative Hopfield
    attention (``TrueHopfieldLayer``) and whose residual connections are
    gated. Each layer computes

        h <- g_hop * h + Hopfield(LayerNorm(h))
        h <- g_ffn * h + FFN(LayerNorm(h))

    followed by mean pooling and a linear classifier. ``gate_target`` selects
    which residuals use ``gate_init`` / ``gate_mode``; the other residuals use
    a fixed gate at the clamp limit (0.9999), i.e. an ungated residual.
    """

    def __init__(
        self,
        task_config: dict,
        model_config: dict,
        gate_init: float = 1.0,
        gate_mode: str = 'fixed',
        gate_target: str = 'hopfield',
    ):
        super().__init__()
        assert gate_target in ('hopfield', 'ffn', 'both')

        self.num_labels = task_config['num_labels']
        self.use_segment_embeddings = model_config.get('segment_embeddings', True)
        self.gate_init = gate_init
        self.gate_mode = gate_mode
        self.gate_target = gate_target
        self.num_layers = model_config['num_hopfield_layers']

        self.token_embeddings = nn.Embedding(
            model_config['vocab_size'], model_config['hidden_size'], padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            model_config['max_position_embeddings'], model_config['hidden_size']
        )
        if self.use_segment_embeddings:
            self.segment_embeddings = nn.Embedding(
                model_config.get('type_vocab_size', 2), model_config['hidden_size']
            )
        self.embed_norm = nn.LayerNorm(model_config['hidden_size'])
        self.dropout = nn.Dropout(model_config['dropout'])

        self.hopfield_layers = nn.ModuleList(
            [TrueHopfieldLayer(model_config) for _ in range(self.num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(model_config['hidden_size']) for _ in range(self.num_layers)]
        )
        self.ffn_layers = nn.ModuleList(
            [FeedForward(model_config) for _ in range(self.num_layers)]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(model_config['hidden_size']) for _ in range(self.num_layers)]
        )

        hop_gate_init = gate_init if gate_target in ('hopfield', 'both') else 1.0
        ffn_gate_init = gate_init if gate_target in ('ffn',     'both') else 1.0
        hop_gate_mode = gate_mode if gate_target in ('hopfield', 'both') else 'fixed'
        ffn_gate_mode = gate_mode if gate_target in ('ffn',     'both') else 'fixed'

        self.hopfield_gates = nn.ModuleList(
            [ResidualGate(gate_init=hop_gate_init, gate_mode=hop_gate_mode)
             for _ in range(self.num_layers)]
        )
        self.ffn_gates = nn.ModuleList(
            [ResidualGate(gate_init=ffn_gate_init, gate_mode=ffn_gate_mode)
             for _ in range(self.num_layers)]
        )

        self.hopfield_pool = HopfieldPooling()
        self.classifier = nn.Linear(model_config['hidden_size'], self.num_labels)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        B, L = input_ids.shape
        tok = self.token_embeddings(input_ids)
        pos_ids = torch.arange(L, dtype=torch.long, device=input_ids.device)
        pos = self.position_embeddings(pos_ids.unsqueeze(0).expand(B, -1))
        emb = tok + pos
        if self.use_segment_embeddings:
            if token_type_ids is None:
                token_type_ids = torch.zeros(B, L, dtype=torch.long, device=input_ids.device)
            emb = emb + self.segment_embeddings(token_type_ids)
        return self.embed_norm(emb)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, embeddings=None):
        """
        Return ``(logits, layer_iterations)``.

        If ``embeddings`` is given, it is used as the (already normalised)
        encoder input and ``input_ids`` is ignored; this is how the PGD attack
        feeds perturbed embeddings. ``layer_iterations`` holds one
        ``(iterations, converged)`` tuple per Hopfield layer.
        """
        if embeddings is not None:
            hidden = embeddings
        else:
            hidden = self.get_embeddings(input_ids, attention_mask, token_type_ids)
            hidden = self.dropout(hidden)

        layer_iterations = []

        for hopfield, ln, ffn, ffn_ln, hop_gate, ffn_gate in zip(
            self.hopfield_layers, self.layer_norms,
            self.ffn_layers, self.ffn_norms,
            self.hopfield_gates, self.ffn_gates,
        ):
            residual = hidden
            hopfield_out, iters, converged = hopfield(ln(hidden), attention_mask)
            layer_iterations.append((iters, converged))
            hidden = hop_gate(residual, hopfield_out)

            residual = hidden
            ffn_out = ffn(ffn_ln(hidden))
            hidden = ffn_gate(residual, ffn_out)

        pooled = self.hopfield_pool(hidden, attention_mask)
        logits = self.classifier(pooled)
        return logits, layer_iterations

    def get_gate_values(self) -> dict:
        return {
            **{f'l{i}_hop_gate': g.gate_value for i, g in enumerate(self.hopfield_gates)},
            **{f'l{i}_ffn_gate': g.gate_value for i, g in enumerate(self.ffn_gates)},
        }


# ============================================================================
# Spectral radius tracker via Jacobian power iteration
# ============================================================================

def get_spectral_radius(hopfield_layer, gate_layer, ln_layer, x, attention_mask, num_iters=10):
    """
    Estimate the spectral radius of the Jacobian of one gated Hopfield
    sublayer, ``h -> gate(h, hopfield(LayerNorm(h)))``, at ``x`` by power
    iteration with Jacobian-vector products. Returns the mean ratio
    ``||J v|| / ||v||`` over the batch and sequence positions.
    """
    def step_fn(h):
        hop_out, _, _ = hopfield_layer(ln_layer(h), attention_mask)
        return gate_layer(h, hop_out)

    v = torch.randn_like(x)
    v = v / (torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-8)

    for _ in range(num_iters):
        _, Jv = torch.autograd.functional.jvp(step_fn, x, v=v, strict=False)
        v = Jv / (torch.norm(Jv, p=2, dim=-1, keepdim=True) + 1e-8)

    _, Jv = torch.autograd.functional.jvp(step_fn, x, v=v, strict=False)
    rho = torch.norm(Jv, p=2, dim=-1) / (torch.norm(v, p=2, dim=-1) + 1e-8)
    return rho.mean().item()


# ============================================================================
# PGD attack and evaluation
# ============================================================================

def pgd_attack(
    model, embeddings, attention_mask, labels, epsilon, alpha, pgd_steps,
    norm='l2', track_basin_ridge=False, track_grad_decay=False, is_regression=False,
):
    """
    Projected gradient descent in the input embedding space.

    Maximises the task loss within an L2 ball (per-example norm over all
    tokens and dimensions) or an L-infinity ball of radius ``epsilon``, taking
    ``pgd_steps`` steps of size ``alpha`` from a small random start.

    Returns ``(adv_embeddings, rho_trajectory, grad_norm_trajectory)``; the
    trajectories are empty unless ``track_basin_ridge`` / ``track_grad_decay``
    are set.
    """
    model.eval()
    delta = torch.zeros_like(embeddings)

    if norm == 'l2':
        noise = torch.randn_like(embeddings)
        noise_norm = noise.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
        delta = epsilon * 0.1 * noise / noise_norm
    elif norm == 'linf':
        delta = torch.empty_like(embeddings).uniform_(-epsilon, epsilon)

    delta = delta.detach()
    rho_trajectory = []
    grad_norm_trajectory = []

    for step in range(pgd_steps):
        delta.requires_grad_(True)
        adv_emb = embeddings + delta

        if track_basin_ridge:
            local_rho = get_spectral_radius(
                model.hopfield_layers[0],
                model.hopfield_gates[0],
                model.layer_norms[0],
                adv_emb.detach(),
                attention_mask,
            )
            rho_trajectory.append(local_rho)

        logits, _ = model(attention_mask=attention_mask, embeddings=adv_emb)

        if is_regression:
            loss = F.mse_loss(logits.squeeze(), labels)
        else:
            loss = F.cross_entropy(logits, labels)

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.detach()

            if track_grad_decay:
                grad_norm_trajectory.append(grad.norm(p=2).item())

            if norm == 'l2':
                g_norm = grad.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                delta = delta + alpha * grad / g_norm
                d_norm = delta.norm(p=2, dim=(1, 2), keepdim=True).clamp(min=1e-8)
                delta = delta * torch.clamp(epsilon / d_norm, max=1.0)
            elif norm == 'linf':
                delta = delta + alpha * grad.sign()
                delta = delta.clamp(-epsilon, epsilon)

        delta = delta.detach()

    return (embeddings + delta).detach(), rho_trajectory, grad_norm_trajectory


def evaluate_pgd(
    model, val_loader, task_name, eval_epsilons, pgd_steps, pgd_alpha_factor,
    pgd_norm, device, track_basin_ridge=False, track_grad_decay=False,
):
    """
    Evaluate robustness to PGD attacks for each budget in ``eval_epsilons``.

    For every epsilon, reports the clean and adversarial task metrics
    (``clean_*`` / ``adv_*``), the attack success rate (fraction of correctly
    classified examples whose prediction is flipped) and the average number
    of Hopfield iterations on clean and adversarial inputs. Also estimates
    the clean spectral radius of the first gated Hopfield sublayer
    (``global_clean_rho``) on the first three validation batches.
    """
    model.eval()
    results = {}
    is_regression = TASK_CONFIGS[task_name]['num_labels'] == 1

    print("\nEstimating global spectral radius (rho) on clean data...")
    clean_rhos = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            clean_emb = model.get_embeddings(input_ids, attention_mask, token_type_ids)
            r = get_spectral_radius(
                model.hopfield_layers[0],
                model.hopfield_gates[0],
                model.layer_norms[0],
                clean_emb,
                attention_mask,
            )
            clean_rhos.append(r)

    global_clean_rho = sum(clean_rhos) / len(clean_rhos) if clean_rhos else 1.0
    print(f"-> Global clean spectral radius rho(J_F) ~= {global_clean_rho:.4f}")

    for epsilon in eval_epsilons:
        alpha = pgd_alpha_factor * epsilon
        print(f"\n  eps={epsilon:.3f}  alpha={alpha:.4f}  steps={pgd_steps}  norm={pgd_norm}")

        clean_preds_all, adv_preds_all, labels_all = [], [], []
        clean_correct_all, adv_correct_all = [], []
        iters_clean_all, iters_adv_all = [], []
        first_batch_rho_traj = []
        first_batch_grad_traj = []

        for b_idx, batch in enumerate(tqdm(val_loader, desc=f"  PGD eps={epsilon}", leave=False)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with torch.no_grad():
                clean_emb = model.get_embeddings(input_ids, attention_mask, token_type_ids)
                clean_logits, clean_layer_iters = model(
                    attention_mask=attention_mask, embeddings=clean_emb
                )

            if is_regression:
                clean_preds = clean_logits.squeeze()
            else:
                clean_preds = clean_logits.argmax(dim=-1)
                clean_correct_all.append((clean_preds == labels).cpu())

            iters_clean_all.append(
                sum(it for (it, _) in clean_layer_iters) / len(clean_layer_iters)
            )

            track_this = (b_idx == 0)
            adv_emb, rho_traj, grad_traj = pgd_attack(
                model, clean_emb.detach(), attention_mask, labels,
                epsilon=epsilon, alpha=alpha, pgd_steps=pgd_steps, norm=pgd_norm,
                track_basin_ridge=(track_basin_ridge and track_this),
                track_grad_decay=(track_grad_decay and track_this),
                is_regression=is_regression,
            )
            if track_this:
                first_batch_rho_traj = rho_traj
                first_batch_grad_traj = grad_traj

            with torch.no_grad():
                adv_logits, adv_layer_iters = model(
                    attention_mask=attention_mask, embeddings=adv_emb
                )

            if is_regression:
                adv_preds = adv_logits.squeeze()
            else:
                adv_preds = adv_logits.argmax(dim=-1)
                adv_correct_all.append((adv_preds == labels).cpu())

            iters_adv_all.append(
                sum(it for (it, _) in adv_layer_iters) / len(adv_layer_iters)
            )

            clean_preds_all.extend(clean_preds.cpu().numpy())
            adv_preds_all.extend(adv_preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

        val_meta = getattr(val_loader.dataset, 'metadata', None)
        clean_metrics = compute_metrics(
            task_name, clean_preds_all, labels_all, metadata=val_meta, include_accuracy=True
        )
        adv_metrics = compute_metrics(
            task_name, adv_preds_all, labels_all, metadata=val_meta, include_accuracy=True
        )

        if not is_regression:
            clean_correct_t = torch.cat(clean_correct_all)
            adv_correct_t = torch.cat(adv_correct_all)
            n_clean_correct = clean_correct_t.sum().item()
            asr = (
                (clean_correct_t & ~adv_correct_t).float().sum().item() / n_clean_correct
                if n_clean_correct > 0 else 0.0
            )
        else:
            asr = 0.0

        avg_iters_clean = sum(iters_clean_all) / len(iters_clean_all)
        avg_iters_adv = sum(iters_adv_all) / len(iters_adv_all)

        eps_res = {
            'asr': asr,
            'avg_iters_clean': avg_iters_clean,
            'avg_iters_adv': avg_iters_adv,
            'iter_gap': avg_iters_adv - avg_iters_clean,
            'rho_trajectory': first_batch_rho_traj if first_batch_rho_traj else None,
            'grad_norm_trajectory': first_batch_grad_traj if first_batch_grad_traj else None,
        }
        for k, v in clean_metrics.items():
            eps_res[f'clean_{k}'] = v
        for k, v in adv_metrics.items():
            eps_res[f'adv_{k}'] = v

        results[epsilon] = eps_res

        primary_key = list(clean_metrics.keys())[0]
        print(
            f"    clean_{primary_key}={clean_metrics[primary_key]:.4f}"
            f"  adv_{primary_key}={adv_metrics[primary_key]:.4f}"
            f"  ASR={asr:.4f}"
            f"  iter_gap={eps_res['iter_gap']:+.2f}"
        )

    results['global_clean_rho'] = global_clean_rho
    return results


# ============================================================================
# Training
# ============================================================================

def train_model(
    model, train_loader, val_loader, task_name, model_path, epochs, device, model_config
):
    """
    Fine-tune with AdamW and a linear warmup schedule. The checkpoint with the
    best primary validation metric is saved to ``model_path``; training stops
    early after ``model_config['patience']`` epochs without improvement.

    Returns ``(model, history)``, where ``history`` holds the per-epoch
    training curves (loss, validation metric, gate values and Hopfield
    iteration / convergence statistics).
    """
    model = model.to(device)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    is_regression = TASK_CONFIGS[task_name]['num_labels'] == 1
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    best_metric = -float('inf')
    patience_counter = 0
    patience = model_config['patience']

    history = {
        'epoch':                     [],
        'train_loss':                [],
        'val_metric':                [],
        'val_metric_name':           None,
        'gate_values':               [],
        'train_avg_iters':           [],
        'train_conv_rate':           [],
        'train_per_layer_avg_iters': [],
        'val_avg_iters':             [],
        'val_conv_rate':             [],
        'val_per_layer_avg_iters':   [],
        'stopped_epoch':             None,
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_layers = model.num_layers
        train_iters_per_layer = [[] for _ in range(n_layers)]
        train_conv_per_layer  = [[] for _ in range(n_layers)]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            logits, layer_iters = model(input_ids, attention_mask, token_type_ids)

            for li, (iters, converged) in enumerate(layer_iters):
                train_iters_per_layer[li].append(iters)
                train_conv_per_layer[li].append(int(converged))

            if is_regression:
                loss = criterion(logits.squeeze(), labels)
            else:
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_preds, val_labels_list = [], []
        val_iters_per_layer = [[] for _ in range(n_layers)]
        val_conv_per_layer  = [[] for _ in range(n_layers)]

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                logits, layer_iters = model(input_ids, attention_mask, token_type_ids)

                for li, (iters, converged) in enumerate(layer_iters):
                    val_iters_per_layer[li].append(iters)
                    val_conv_per_layer[li].append(int(converged))

                if is_regression:
                    preds = logits.squeeze()
                else:
                    preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_meta = getattr(val_loader.dataset, 'metadata', None)
        val_metrics = compute_metrics(task_name, val_preds, val_labels_list, metadata=val_meta)

        primary = list(val_metrics.keys())[0]
        current = val_metrics[primary]

        def _layer_mean(per_layer):
            return [float(np.mean(lst)) if lst else 0.0 for lst in per_layer]

        def _global_mean(per_layer):
            all_vals = [v for lst in per_layer for v in lst]
            return float(np.mean(all_vals)) if all_vals else 0.0

        tr_layer_avg  = _layer_mean(train_iters_per_layer)
        tr_layer_conv = _layer_mean(train_conv_per_layer)
        vl_layer_avg  = _layer_mean(val_iters_per_layer)
        vl_layer_conv = _layer_mean(val_conv_per_layer)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_metric'].append(float(current))
        if history['val_metric_name'] is None:
            history['val_metric_name'] = primary
        history['gate_values'].append(model.get_gate_values())
        history['train_avg_iters'].append(_global_mean(train_iters_per_layer))
        history['train_conv_rate'].append(float(np.mean(tr_layer_conv)))
        history['train_per_layer_avg_iters'].append(tr_layer_avg)
        history['val_avg_iters'].append(_global_mean(val_iters_per_layer))
        history['val_conv_rate'].append(float(np.mean(vl_layer_conv)))
        history['val_per_layer_avg_iters'].append(vl_layer_avg)

        print(
            f"\nEpoch {epoch+1}"
            f"  train_loss={history['train_loss'][-1]:.4f}"
            f"  val_{primary}={current:.4f}"
            f"  train_avg_iters={history['train_avg_iters'][-1]:.1f}"
            f"  train_conv={history['train_conv_rate'][-1]:.2%}"
            f"  val_avg_iters={history['val_avg_iters'][-1]:.1f}"
        )

        if current > best_metric:
            best_metric = current
            patience_counter = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'epoch': epoch,
                    'gate_values': model.get_gate_values(),
                },
                model_path,
            )
            print(f"  [best] {primary}={best_metric:.4f} -> {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {epoch+1} epochs")
                history['stopped_epoch'] = epoch + 1
                break

    return model, history


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Agatha (GatedHopfieldModel): training and PGD robustness evaluation'
    )

    # Architecture
    parser.add_argument('--hidden_size',       type=int,   default=DEFAULT_MODEL_CONFIG['hidden_size'])
    parser.add_argument('--num_heads',         type=int,   default=DEFAULT_MODEL_CONFIG['num_heads'])
    parser.add_argument('--num_layers',        type=int,   default=DEFAULT_MODEL_CONFIG['num_hopfield_layers'])
    parser.add_argument('--intermediate_size', type=int,   default=DEFAULT_MODEL_CONFIG['intermediate_size'])
    parser.add_argument('--beta',              type=float, default=15.0,
                        help='Inverse temperature of the Hopfield update (default: 15.0)')
    parser.add_argument('--max_iterations',    type=int,   default=50,
                        help='Maximum Hopfield iterations per layer (default: 50)')
    parser.add_argument('--dropout',           type=float, default=DEFAULT_MODEL_CONFIG['dropout'])
    parser.add_argument('--patience',          type=int,   default=DEFAULT_MODEL_CONFIG['patience'],
                        help='Early-stopping patience in epochs (default: 5)')
    parser.add_argument('--segment_embeddings', type=str,  default='True',
                        help='Use segment (token type) embeddings (default: True)')

    # Gating
    parser.add_argument('--gate_mode',   type=str,   default='fixed', choices=['fixed', 'learned'],
                        help='Keep the gates fixed at --gate_init or learn them (default: fixed)')
    parser.add_argument('--gate_init',   type=float, default=1.0,
                        help='Initial gate value g in (0, 1), clamped to [1e-4, 0.9999] (default: 1.0)')
    parser.add_argument('--gate_target', type=str,   default='hopfield', choices=['hopfield', 'ffn', 'both'],
                        help='Residual connections to gate (default: hopfield)')

    # Pretrained initialisation
    parser.add_argument(
        '--pretrained', type=str, default='False',
        help=(
            'Initialise from a pretrained BERT checkpoint before fine-tuning. '
            'Checkpoint is selected automatically from (num_layers, num_heads, hidden_size): '
            'L2/H2/dim128 -> google/bert_uncased_L-2_H-128_A-2, '
            'L4/H4/dim256 -> google/bert_uncased_L-4_H-256_A-4, '
            'L12/H12/dim768 -> bert-base-uncased. '
            '(default: False)'
        ),
    )
    parser.add_argument(
        '--pretrained_checkpoint', type=str, default=None,
        help=(
            'Override the auto-selected checkpoint with an explicit Hugging Face model ID. '
            'Only used when --pretrained True.'
        ),
    )

    # PGD evaluation
    parser.add_argument('--pgd_steps',        type=int,   default=20,
                        help='Number of PGD steps (default: 20)')
    parser.add_argument('--pgd_alpha_factor',  type=float, default=0.3,
                        help='PGD step size as a fraction of epsilon (default: 0.3)')
    parser.add_argument('--pgd_norm',          type=str,   default='l2', choices=['l2', 'linf'],
                        help='Norm of the perturbation ball (default: l2)')
    parser.add_argument('--eval_epsilons',     type=str,   default='0.05,0.5,1.0,2.0,3.0,5.0',
                        help='Comma-separated perturbation budgets (default: 0.05,0.5,1.0,2.0,3.0,5.0)')
    parser.add_argument('--track_basin_ridge', type=str,   default='False',
                        help='Track the Jacobian spectral radius along the PGD path of the first batch (default: False)')
    parser.add_argument('--track_grad_decay',  type=str,   default='False',
                        help='Track the input-gradient norm per PGD step for the first batch (default: False)')

    # Task / training
    parser.add_argument('--task',             default='sst2', choices=list(TASK_CONFIGS.keys()))
    parser.add_argument('--epochs',           type=int, default=10)
    parser.add_argument('--batch_size',       type=int, default=32)
    parser.add_argument('--eval_batch_size',  type=int, default=64)
    parser.add_argument('--seed',             type=int, default=42)
    parser.add_argument('--train',            type=str, default='True',
                        help='Train before evaluating (default: True). With False, evaluate an existing checkpoint')
    parser.add_argument('--max_length',       type=int, default=128)
    parser.add_argument('--device',           type=str, default='cuda',
                        help='cuda, cuda:0, ... or cpu; falls back to CPU if CUDA is unavailable (default: cuda)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optionally subsample the training set (useful for quick tests)')

    # I/O
    parser.add_argument('--model_dir',       default='./models_gated',
                        help='Directory for checkpoints (default: ./models_gated)')
    parser.add_argument('--trained_model',   type=str, default=None,
                        help='Checkpoint (.pt) to evaluate when --train False')
    parser.add_argument('--out_dir',         type=str, default=None,
                        help='Directory for result JSON files (default: --model_dir)')

    args = parser.parse_args()

    eval_epsilons    = [float(e) for e in args.eval_epsilons.split(',')]
    do_track_basin   = args.track_basin_ridge.lower() in ('true', '1', 'yes')
    do_track_grad    = args.track_grad_decay.lower()  in ('true', '1', 'yes')
    do_train         = args.train.lower()             in ('true', '1', 'yes')
    do_pretrained    = args.pretrained.lower()        in ('true', '1', 'yes')

    # ------------------------------------------------------------------ config
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update({
        'hidden_size':         args.hidden_size,
        'num_heads':           args.num_heads,
        'num_hopfield_layers': args.num_layers,
        'intermediate_size':   args.intermediate_size,
        'beta':                args.beta,
        'max_iterations':      args.max_iterations,
        'dropout':             args.dropout,
        'patience':            args.patience,
        'segment_embeddings':  args.segment_embeddings.lower() in ('true', '1', 'yes'),
    })

    # ------------------------------------------------------------------ tag
    pretrained_suffix = ''
    if do_pretrained:
        ckpt = args.pretrained_checkpoint or get_pretrained_checkpoint(model_config)
        if ckpt is not None:
            pretrained_suffix = '_' + ckpt.split('/')[-1]

    tag = (
        f"{args.task}_L{args.num_layers}_H{args.num_heads}"
        f"_b{args.beta}_g{args.gate_init}_iter{args.max_iterations}"
        f"{pretrained_suffix}_seed{args.seed}"
    )

    os.makedirs(args.model_dir, exist_ok=True)
    out_dir = args.out_dir if args.out_dir else args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    default_model_path = os.path.join(args.model_dir, f"{tag}.pt")

    # ------------------------------------------------------------------ device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    seed_worker = set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # ------------------------------------------------------------------ banner
    print(f"\n{'='*60}")
    print(f"Experiment: {tag}")
    print(f"  gate_mode={args.gate_mode}  gate_init={args.gate_init}  gate_target={args.gate_target}")
    print(f"  max_iter={args.max_iterations}  beta={args.beta}")
    print(f"  pretrained={do_pretrained}  checkpoint={args.pretrained_checkpoint or 'auto'}")
    print(f"  track_basin_ridge={do_track_basin}  track_grad_decay={do_track_grad}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset, val_dataset = load_data(
        args.task, tokenizer, args.max_length, args.max_train_samples
    )
    train_loader = (
        DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=g,
        )
        if train_dataset else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g,
    )

    # ------------------------------------------------------------------ model
    model = GatedHopfieldModel(
        TASK_CONFIGS[args.task], model_config,
        gate_init=args.gate_init, gate_mode=args.gate_mode, gate_target=args.gate_target,
    )

    if do_pretrained:
        model = load_pretrained_bert_weights(
            model, model_config, checkpoint=args.pretrained_checkpoint or None
        )

    # ------------------------------------------------------------------ train / load
    if do_train:
        model, training_history = train_model(
            model, train_loader, val_loader, args.task,
            default_model_path, args.epochs, device, model_config,
        )
        load_path = default_model_path

        curves_file = os.path.join(out_dir, f"training_curves_{tag}.json")
        with open(curves_file, 'w') as f:
            json.dump(
                {'tag': tag, 'config': vars(args), 'history': training_history},
                f, indent=2,
            )
        print(f"\nTraining curves saved -> {curves_file}")
    else:
        load_path = args.trained_model if args.trained_model else default_model_path

    if os.path.exists(load_path):
        try:
            ckpt = torch.load(load_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"\nLoaded model from {load_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load weights from {load_path}. "
                f"Make sure you passed a .pt file, not a .json. Error: {e}"
            ) from e
    else:
        if not do_train and args.trained_model:
            raise FileNotFoundError(
                f"--trained_model was set to '{load_path}' but the file does not exist."
            )
        else:
            print(
                f"\nWARNING: No checkpoint found at {load_path}. "
                f"Evaluating with current weights."
            )

    model = model.to(device)

    # ------------------------------------------------------------------ PGD eval
    print(
        f"\n{'='*60}\n"
        f"PGD Evaluation  (steps={args.pgd_steps}, norm={args.pgd_norm})\n"
        f"  epsilons: {eval_epsilons}\n"
        f"{'='*60}"
    )
    pgd_results = evaluate_pgd(
        model, val_loader, args.task, eval_epsilons,
        args.pgd_steps, args.pgd_alpha_factor, args.pgd_norm, device,
        track_basin_ridge=do_track_basin, track_grad_decay=do_track_grad,
    )

    output = {
        'tag': tag,
        'config': vars(args),
        'final_gate_values': model.get_gate_values(),
        'pgd_results': {str(k): v for k, v in pgd_results.items() if k != 'global_clean_rho'},
        'global_clean_rho': pgd_results.get('global_clean_rho', None),
    }
    out_file = os.path.join(out_dir, f"{tag}.json")
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {out_file}")


if __name__ == '__main__':
    main()
