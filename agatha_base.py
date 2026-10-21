"""
Iterative Hopfield encoder and the ungated baseline model.

This module provides the building blocks shared with Agatha (``agatha.py``):
the multi-head iterative Hopfield layer, the feed-forward block, mean pooling,
the default model configuration and the pretrained BERT weight loader. It also
defines ``IterativeHopfieldModel``, the baseline that uses standard (ungated)
residual connections.

Run as a script, it trains the baseline and evaluates it under Gaussian noise
injected into the input embeddings:

    python agatha_base.py --task sst2 --train True --pretrained True
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup

from benchmark_dataloader import TASK_CONFIGS, compute_metrics, load_data
from noise_utils import add_noise_to_embeddings


# ============================================================================
# Pretrained BERT checkpoint registry
#
# Maps (num_hopfield_layers, num_heads, hidden_size) to a Hugging Face
# checkpoint name. This registry is shared by both the gated and base models.
#
#   BERT-tiny  : 2 layers,  2 heads, hidden=128,  intermediate=512
#                google/bert_uncased_L-2_H-128_A-2
#   BERT-mini  : 4 layers,  4 heads, hidden=256,  intermediate=1024
#                google/bert_uncased_L-4_H-256_A-4
#   BERT-base  : 12 layers, 12 heads, hidden=768, intermediate=3072
#                bert-base-uncased
# ============================================================================

PRETRAINED_CHECKPOINT_MAP = {
    (2,  2,  128): 'google/bert_uncased_L-2_H-128_A-2',   # BERT-tiny   (4.4M)
    (4,  4,  256): 'google/bert_uncased_L-4_H-256_A-4',   # BERT-mini  (11.3M)
    (12, 12, 768): 'bert-base-uncased',                    # BERT-base (109M)
}


def get_pretrained_checkpoint(model_config):
    """
    Return the Hugging Face checkpoint string for the given model config,
    or None if no registered entry matches.
    """
    key = (
        model_config['num_hopfield_layers'],
        model_config['num_heads'],
        model_config['hidden_size'],
    )
    return PRETRAINED_CHECKPOINT_MAP.get(key, None)


def load_pretrained_bert_weights(model, model_config, checkpoint=None):
    """
    Initialise a Hopfield encoder from a pretrained BERT checkpoint.

    Works for both ``IterativeHopfieldModel`` (this module) and
    ``GatedHopfieldModel`` (agatha.py).

    Weight mapping
    --------------
    Embeddings
        word_embeddings            ->  token_embeddings
        position_embeddings        ->  position_embeddings
        token_type_embeddings      ->  segment_embeddings
        embeddings.LayerNorm       ->  embed_norm (gated) / layer_norm (base)

    Per encoder layer i
        attention.self.query       ->  hopfield_layers[i].W_query
        attention.self.key         ->  hopfield_layers[i].W_key
        attention.self.value       ->  hopfield_layers[i].W_value
        attention.output.dense     ->  hopfield_layers[i].W_out
        attention.output.LayerNorm ->  layer_norms[i]   (pre-norm warm-start)
        intermediate.dense         ->  ffn_layers[i].linear1
        output.dense               ->  ffn_layers[i].linear2
        output.LayerNorm           ->  ffn_norms[i]     (pre-norm warm-start)

    Not transferred
        BERT pooler.dense          (the models use mean pooling)
        classifier head            (task-specific, randomly initialised)
        gate parameters            (Agatha only; set by --gate_init / --gate_mode)

    Note on LayerNorm position
    --------------------------
    BERT is post-norm: LayerNorm(x + sublayer(x)).
    The Hopfield encoders are pre-norm: sublayer(LayerNorm(x)) plus a standard
    (base) or gated (Agatha) residual. LayerNorm weights are transferred as a
    warm-start and adapt during fine-tuning.

    Parameters
    ----------
    model        : IterativeHopfieldModel or GatedHopfieldModel
                   (constructed, before .to(device))
    model_config : dict with num_hopfield_layers, num_heads, hidden_size, etc.
    checkpoint   : Hugging Face model ID, or None to auto-detect from config.

    Raises
    ------
    ValueError   : if checkpoint is None and no registry entry matches config.
    RuntimeError : if a weight tensor has an incompatible shape.
    """
    if checkpoint is None:
        checkpoint = get_pretrained_checkpoint(model_config)
        if checkpoint is None:
            supported = ', '.join(
                f'L{layers}/H{heads}/dim{dim}={ckpt}'
                for (layers, heads, dim), ckpt in PRETRAINED_CHECKPOINT_MAP.items()
            )
            raise ValueError(
                f"No pretrained checkpoint registered for "
                f"L={model_config['num_hopfield_layers']}, "
                f"H={model_config['num_heads']}, "
                f"hidden={model_config['hidden_size']}. "
                f"Supported configs: {supported}. "
                f"Pass --pretrained_checkpoint to override."
            )

    print(f"\nLoading pretrained weights from: {checkpoint}")
    bert = BertModel.from_pretrained(checkpoint)
    sd = bert.state_dict()
    n_layers = model_config['num_hopfield_layers']

    # The embedding LayerNorm is `embed_norm` in GatedHopfieldModel and
    # `layer_norm` in IterativeHopfieldModel.
    embed_norm = model.embed_norm if hasattr(model, 'embed_norm') else model.layer_norm

    # -- Embeddings -------------------------------------------------------
    model.token_embeddings.weight.data.copy_(
        sd['embeddings.word_embeddings.weight']
    )
    model.position_embeddings.weight.data.copy_(
        sd['embeddings.position_embeddings.weight']
    )
    if model.use_segment_embeddings:
        model.segment_embeddings.weight.data.copy_(
            sd['embeddings.token_type_embeddings.weight']
        )
    embed_norm.weight.data.copy_(sd['embeddings.LayerNorm.weight'])
    embed_norm.bias.data.copy_(sd['embeddings.LayerNorm.bias'])

    # -- Encoder layers ---------------------------------------------------
    for i in range(n_layers):
        p = f'encoder.layer.{i}'
        hopfield = model.hopfield_layers[i]
        for target, bert_name in [
            (hopfield.W_query,             'attention.self.query'),
            (hopfield.W_key,               'attention.self.key'),
            (hopfield.W_value,             'attention.self.value'),
            (hopfield.W_out,               'attention.output.dense'),
            (model.layer_norms[i],         'attention.output.LayerNorm'),
            (model.ffn_layers[i].linear1,  'intermediate.dense'),
            (model.ffn_layers[i].linear2,  'output.dense'),
            (model.ffn_norms[i],           'output.LayerNorm'),
        ]:
            target.weight.data.copy_(sd[f'{p}.{bert_name}.weight'])
            target.bias.data.copy_(sd[f'{p}.{bert_name}.bias'])

    del bert, sd

    print(f"Transferred {n_layers} encoder layers from '{checkpoint}'.")
    print("Classifier head (and gate parameters, if any) keep their initial values.")
    return model


# ============================================================================
# AdvGLUE helper functions
# ============================================================================

def is_advglue_task(task_name):
    return task_name in TASK_CONFIGS and TASK_CONFIGS[task_name].get('benchmark') == 'advglue'


def get_base_task(task_name):
    """Return the GLUE task an AdvGLUE task is derived from (or the task itself)."""
    if is_advglue_task(task_name):
        return TASK_CONFIGS[task_name].get('base_task', task_name)
    return task_name


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed):
    """
    Seed Python, NumPy and PyTorch, and make cuDNN deterministic.

    Returns a ``worker_init_fn`` for DataLoader workers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


# ============================================================================
# Default model configuration
# ============================================================================

DEFAULT_MODEL_CONFIG = {
    'hidden_size': 128,
    'num_heads': 2,
    'num_hopfield_layers': 2,
    'intermediate_size': 512,
    'vocab_size': 30522,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
    'segment_embeddings': True,
    'beta': 50.0,
    'max_iterations': 50,
    'convergence_threshold': 1e-4,
    'dropout': 0.3,
    'patience': 5
}


def get_model_tag(model_config):
    """Generate a tag like L2_H2 from the model config."""
    return f"L{model_config['num_hopfield_layers']}_H{model_config['num_heads']}"


# ============================================================================
# Multi-head Hopfield layer
# ============================================================================

class TrueHopfieldLayer(nn.Module):
    """
    Multi-head iterative Hopfield attention.

    Starting from the query projections, each head repeatedly applies the
    modern Hopfield update

        state <- softmax(beta * state @ K^T / sqrt(head_dim)) @ K

    until the relative change of the state drops below
    ``convergence_threshold`` or ``max_iterations`` is reached. The retrieved
    state then attends over the values, and the heads are merged and projected
    as in standard multi-head attention.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // config['num_heads']
        self.beta = config['beta']
        self.max_iterations = config['max_iterations']
        self.convergence_threshold = config['convergence_threshold']

        self.W_query = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_key = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_value = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.W_out = nn.Linear(config['hidden_size'], config['hidden_size'])

        self.dropout = nn.Dropout(config['dropout'])

    def _split_heads(self, x):
        """(batch, seq, hidden) -> (batch, num_heads, seq, head_dim)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """(batch, num_heads, seq, head_dim) -> (batch, seq, hidden)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.hidden_size)

    def hopfield_update(self, query, keys):
        """Single Hopfield update step (operates on per-head tensors)."""
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = self.beta * scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        updated_query = torch.matmul(attn_weights, keys)
        return updated_query, attn_weights

    def forward(self, x, attention_mask=None):
        """
        Run the Hopfield iteration.

        Returns ``(output, iterations, converged)``, where ``iterations`` is the
        number of update steps taken and ``converged`` tells whether the
        convergence threshold was reached before ``max_iterations``.
        """
        query = self._split_heads(self.W_query(x))
        keys = self._split_heads(self.W_key(x))
        values = self._split_heads(self.W_value(x))

        state = query.clone()
        iterations = 0
        converged = False

        for iteration in range(self.max_iterations):
            prev_state = state.clone()
            state, attn_weights = self.hopfield_update(state, keys)
            iterations = iteration + 1

            delta = torch.norm(state - prev_state) / (torch.norm(prev_state) + 1e-8)
            if delta < self.convergence_threshold:
                converged = True
                break

        # Final attention computation using the converged state
        scores = torch.matmul(state, keys.transpose(-2, -1))
        scores = self.beta * scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values)

        # Merge heads and project
        output = self._merge_heads(output)
        output = self.W_out(output)
        output = self.dropout(output)

        return output, iterations, converged


# ============================================================================
# Model architecture
# ============================================================================

class FeedForward(nn.Module):
    """Standard FFN with GELU activation."""

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.linear2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class HopfieldPooling(nn.Module):
    """Mean pooling over the non-padding tokens."""

    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = summed / counts
        return pooled


class IterativeHopfieldModel(nn.Module):
    """
    Ungated baseline: a pre-norm encoder whose self-attention is replaced by
    ``TrueHopfieldLayer``, with standard residual connections, mean pooling
    and a linear classification head.
    """

    def __init__(self, task_config, model_config):
        super().__init__()
        self.num_labels = task_config['num_labels']
        self.use_segment_embeddings = model_config.get('segment_embeddings', True)

        # Embeddings
        self.token_embeddings = nn.Embedding(
            model_config['vocab_size'],
            model_config['hidden_size'],
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            model_config['max_position_embeddings'],
            model_config['hidden_size']
        )

        if self.use_segment_embeddings:
            self.segment_embeddings = nn.Embedding(
                model_config.get('type_vocab_size', 2),
                model_config['hidden_size']
            )

        self.layer_norm = nn.LayerNorm(model_config['hidden_size'])
        self.dropout = nn.Dropout(model_config['dropout'])

        # Hopfield layers
        self.hopfield_layers = nn.ModuleList([
            TrueHopfieldLayer(model_config)
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(model_config['hidden_size'])
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.ffn_layers = nn.ModuleList([
            FeedForward(model_config)
            for _ in range(model_config['num_hopfield_layers'])
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(model_config['hidden_size'])
            for _ in range(model_config['num_hopfield_layers'])
        ])

        # Pooling and classification
        self.hopfield_pool = HopfieldPooling()
        self.classifier = nn.Linear(model_config['hidden_size'], self.num_labels)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_length = input_ids.shape

        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds

        if self.use_segment_embeddings:
            if token_type_ids is None:
                token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.segment_embeddings(token_type_ids)

        embeddings = self.layer_norm(embeddings)

        return embeddings

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, embeddings=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs. Required if embeddings is None.
            attention_mask: Attention mask. Always required.
            token_type_ids: Segment IDs for sentence-pair tasks. Optional.
            embeddings: Pre-computed embeddings. If provided, input_ids is ignored
                        and dropout is skipped (caller controls preprocessing).

        Returns:
            (logits, layer_iterations), where layer_iterations holds one
            (iterations, converged) tuple per Hopfield layer.
        """
        if embeddings is not None:
            hidden_states = embeddings
        else:
            hidden_states = self.get_embeddings(input_ids, attention_mask, token_type_ids)
            hidden_states = self.dropout(hidden_states)

        # Through Hopfield layers
        layer_iterations = []

        for hopfield, norm, ffn, ffn_norm in zip(
            self.hopfield_layers, self.layer_norms,
            self.ffn_layers, self.ffn_norms
        ):
            # Hopfield attention with residual
            residual = hidden_states
            hidden_states = norm(hidden_states)
            hopfield_out, iterations, converged = hopfield(hidden_states, attention_mask)
            layer_iterations.append((iterations, converged))
            hidden_states = residual + hopfield_out

            # FFN with residual
            residual = hidden_states
            hidden_states = ffn_norm(hidden_states)
            ffn_out = ffn(hidden_states)
            hidden_states = residual + ffn_out

        # Pool and classify
        pooled = self.hopfield_pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        return logits, layer_iterations

    def print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\nModel Architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Hopfield layers: {len(self.hopfield_layers)}")
        print(f"  Attention heads: {self.hopfield_layers[0].num_heads}")
        print(f"  Head dimension: {self.hopfield_layers[0].head_dim}")
        print(f"  Segment embeddings: {self.use_segment_embeddings}")
        print(f"  Beta: {self.hopfield_layers[0].beta}")
        print(f"  Max iterations: {self.hopfield_layers[0].max_iterations}")
        print(f"  Output labels: {self.num_labels}")


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, task_name, model_tag,
                epochs, device, model_dir, model_config):
    """
    Fine-tune with AdamW and a linear warmup schedule, keeping the checkpoint
    with the best primary validation metric (early stopping on patience).
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if TASK_CONFIGS[task_name]['num_labels'] == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    best_metric = -float('inf')
    patience_counter = 0

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, token_type_ids=token_type_ids)

            if TASK_CONFIGS[task_name]['num_labels'] == 1:
                loss = criterion(logits.squeeze(), labels)
                preds = logits.squeeze()
            else:
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
                logits, _ = model(input_ids, attention_mask, token_type_ids=token_type_ids)

                if TASK_CONFIGS[task_name]['num_labels'] == 1:
                    preds = logits.squeeze()
                else:
                    preds = torch.argmax(logits, dim=-1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_metadata = getattr(train_loader.dataset, 'metadata', None)
        val_metadata = getattr(val_loader.dataset, 'metadata', None)
        train_metrics = compute_metrics(task_name, train_preds, train_labels, metadata=train_metadata)
        val_metrics = compute_metrics(task_name, val_preds, val_labels, metadata=val_metadata)

        print(f"\nEpoch {epoch+1}")
        print(f"  Train loss: {train_loss/len(train_loader):.4f}")
        print(f"  Train metrics: {train_metrics}")
        print(f"  Val metrics: {val_metrics}")

        primary_metric_name = list(val_metrics.keys())[0]
        current_metric = val_metrics[primary_metric_name]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            tag_dir = os.path.join(model_dir, model_tag)
            os.makedirs(tag_dir, exist_ok=True)
            model_path = os.path.join(tag_dir, f'agatha_base_{task_name}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'epoch': epoch,
                'model_config': model_config
            }, model_path)
            print(f"New best model saved! ({primary_metric_name}: {best_metric:.4f}) in path {model_path}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{model_config['patience']})")

            if patience_counter >= model_config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    return model


# ============================================================================
# Evaluation with noise
# ============================================================================

def evaluate_with_noise(model, val_loader, task_name, noise_levels, noise_type, device, seed):
    """
    Evaluate the model with Gaussian noise added to the input embeddings.

    Returns a dict mapping each noise level to the task metrics plus
    aggregate and per-layer Hopfield iteration / convergence statistics.
    """
    model = model.to(device)
    model.eval()

    num_layers = len(model.hopfield_layers)
    results = {}

    for noise_level in noise_levels:
        print(f"\n{'='*70}")
        print(f"Evaluating with noise level: {noise_level} ({noise_type})")
        print(f"{'='*70}")

        all_preds = []
        all_labels = []

        # Per-layer iteration tracking
        iterations_per_layer = [[] for _ in range(num_layers)]
        converged_per_layer = [[] for _ in range(num_layers)]

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Noise={noise_level}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None

                # Get clean embeddings, inject noise, forward
                embeddings = model.get_embeddings(input_ids, attention_mask, token_type_ids)
                noisy_embeddings = add_noise_to_embeddings(embeddings, noise_level, noise_type, seed)

                logits, layer_iters = model(
                    attention_mask=attention_mask,
                    embeddings=noisy_embeddings
                )

                # Record per-layer iterations
                for layer_idx, (iters, conv) in enumerate(layer_iters):
                    iterations_per_layer[layer_idx].append(iters)
                    converged_per_layer[layer_idx].append(conv)

                # Collect predictions
                if TASK_CONFIGS[task_name]['num_labels'] == 1:
                    preds = logits.squeeze()
                else:
                    preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute task metrics
        metadata = getattr(val_loader.dataset, 'metadata', None)
        metrics = compute_metrics(task_name, all_preds, all_labels, metadata=metadata, include_accuracy=True)

        # Aggregate iteration stats
        total_iterations = sum(sum(iters) for iters in iterations_per_layer)
        total_converged = sum(sum(c) for c in converged_per_layer)
        num_batches = len(iterations_per_layer[0])
        num_layer_calls = num_layers * num_batches

        metrics['avg_iterations'] = total_iterations / num_layer_calls if num_layer_calls > 0 else 0
        metrics['convergence_rate'] = total_converged / num_layer_calls if num_layer_calls > 0 else 0

        # Per-layer stats
        for layer_idx in range(num_layers):
            iters = iterations_per_layer[layer_idx]
            convs = converged_per_layer[layer_idx]
            prefix = f'layer{layer_idx}'
            metrics[f'{prefix}_iterations_mean'] = np.mean(iters)
            metrics[f'{prefix}_iterations_std'] = np.std(iters)
            metrics[f'{prefix}_iterations_min'] = int(np.min(iters))
            metrics[f'{prefix}_iterations_max'] = int(np.max(iters))
            metrics[f'{prefix}_iterations_median'] = float(np.median(iters))
            metrics[f'{prefix}_convergence_rate'] = np.mean(convs)

        # Raw per-layer per-batch data for downstream analysis
        metrics['_iterations_per_layer'] = {
            layer_idx: iters for layer_idx, iters in enumerate(iterations_per_layer)
        }

        results[noise_level] = metrics
        print(f"\n{task_name.upper()} Metrics: {metrics}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Iterative Hopfield baseline (ungated): training and noise-robustness evaluation'
    )

    # Model architecture args
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_MODEL_CONFIG['hidden_size'])
    parser.add_argument('--num_heads', type=int, default=DEFAULT_MODEL_CONFIG['num_heads'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_MODEL_CONFIG['num_hopfield_layers'])
    parser.add_argument('--intermediate_size', type=int, default=DEFAULT_MODEL_CONFIG['intermediate_size'])
    parser.add_argument('--beta', type=float, default=15.0,
                        help='Inverse temperature of the Hopfield update (default: 15.0)')
    parser.add_argument('--max_iterations', type=int, default=DEFAULT_MODEL_CONFIG['max_iterations'],
                        help='Maximum Hopfield iterations per layer (default: 50)')
    parser.add_argument('--dropout', type=float, default=DEFAULT_MODEL_CONFIG['dropout'])
    parser.add_argument('--patience', type=int, default=DEFAULT_MODEL_CONFIG['patience'],
                        help='Early-stopping patience in epochs (default: 5)')
    parser.add_argument('--segment_embeddings', type=str, default='True',
                        help='Use segment (token type) embeddings for sentence-pair tasks (default: True). '
                             'Set False for backward compatibility.')

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

    # Task and training args
    parser.add_argument('--task', default='sst2', choices=list(TASK_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size for evaluation. Defaults to --batch_size if not set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=str, default='False',
                        help='Train the model (True) or only evaluate a saved checkpoint (False; default)')
    parser.add_argument('--load', type=str, default='False',
                        help='Load existing model before training (True/False). Use with --train True')
    parser.add_argument('--noise_type', default='absolute', choices=['absolute', 'percentage'])
    parser.add_argument('--model_dir', default='./models',
                        help='Directory for checkpoints and results (default: ./models)')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda, cuda:0, cuda:1, cpu, etc. (default: cuda)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Optionally subsample the training set (useful for quick tests)')

    args = parser.parse_args()
    do_pretrained = args.pretrained.lower() in ('true', '1', 'yes')

    # Build model config from CLI args
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config['hidden_size'] = args.hidden_size
    model_config['num_heads'] = args.num_heads
    model_config['num_hopfield_layers'] = args.num_layers
    model_config['intermediate_size'] = args.intermediate_size
    model_config['beta'] = args.beta
    model_config['max_iterations'] = args.max_iterations
    model_config['dropout'] = args.dropout
    model_config['patience'] = args.patience
    model_config['segment_embeddings'] = args.segment_embeddings.lower() in ['true', '1', 'yes']

    assert model_config['hidden_size'] % model_config['num_heads'] == 0, \
        f"hidden_size ({model_config['hidden_size']}) must be divisible by num_heads ({model_config['num_heads']})"

    model_tag = get_model_tag(model_config)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    # Set seed
    print(f"\n{'='*70}")
    print(f"Setting seed: {args.seed}")
    print(f"{'='*70}\n")
    seed_worker = set_seed(args.seed)

    # Setup device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*70)
    print(f"Configuration [{model_tag}]")
    print("="*70)
    print(f"Hidden Size: {model_config['hidden_size']}")
    print(f"Num Layers: {model_config['num_hopfield_layers']}")
    print(f"Num Heads: {model_config['num_heads']}")
    print(f"Head Dim: {model_config['hidden_size'] // model_config['num_heads']}")
    print(f"FFN Size: {model_config['intermediate_size']}")
    print(f"Segment Embeddings: {model_config['segment_embeddings']}")
    print(f"Beta (Hopfield): {model_config['beta']}")
    print(f"Max Iterations: {model_config['max_iterations']}")
    print(f"Dropout: {model_config['dropout']}")
    print(f"Patience: {model_config['patience']}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("="*70 + "\n")

    # Load tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # AdvGLUE handling
    is_adv = is_advglue_task(args.task)
    base_task = get_base_task(args.task)
    eval_task = args.task

    if is_adv:
        print(f"\n{'='*70}")
        print(f"AdvGLUE Task: {args.task}")
        print(f"Base task for model loading: {base_task}")
        print(f"Evaluation dataset: {eval_task}")
        print(f"{'='*70}\n")

    # Load data
    print(f"Loading {eval_task} dataset...")
    train_dataset, val_dataset = load_data(eval_task, tokenizer, args.max_length, args.max_train_samples)

    g = torch.Generator()
    g.manual_seed(args.seed)

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        train_loader = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    if train_dataset is not None:
        print(f"Train samples: {len(train_dataset)}")
    else:
        print("Train samples: N/A (inference-only task)")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    task_config = TASK_CONFIGS[eval_task]
    model = IterativeHopfieldModel(task_config, model_config)
    model.print_model_info()

    if do_pretrained:
        model = load_pretrained_bert_weights(
            model, model_config,
            checkpoint=args.pretrained_checkpoint or None,
        )

    # Train or load
    train_flag = args.train.lower() in ['true', '1', 'yes']
    load_flag = args.load.lower() in ['true', '1', 'yes']

    if is_adv and train_flag:
        print("\n" + "="*70)
        print("AdvGLUE is an inference-only benchmark.")
        print("="*70)
        return

    if load_flag and train_flag:
        print("\n" + "="*70)
        print("Loading existing model to continue training")
        print("="*70)

        model_task = base_task if is_adv else eval_task
        model_path = os.path.join(args.model_dir, model_tag, f'agatha_base_{model_task}.pt')

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
            print(f"Previous best metrics: {checkpoint.get('metrics', 'N/A')}")
            print("Continuing training from this checkpoint...")
        else:
            print(f"Model not found at {model_path}")
            print("Starting training from scratch instead.")
        print("="*70)

    if train_flag:
        print("\n" + "="*70)
        if load_flag:
            print("Continuing training")
        else:
            print("Training model")
        print("="*70)
        model = train_model(
            model, train_loader, val_loader, eval_task, model_tag,
            args.epochs, device, args.model_dir, model_config
        )

    # Load best model
    print("\n" + "="*70)
    print("Loading best model for evaluation")
    print("="*70)

    model_task = base_task if is_adv else eval_task
    model_path = os.path.join(args.model_dir, model_tag, f'agatha_base_{model_task}.pt')

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {model_path}")
        if is_adv:
            print(f"(Model trained on base task: {base_task})")
        print(f"Best validation metrics: {checkpoint.get('metrics', 'N/A')}")
    else:
        if not train_flag:
            print(f"No saved model found at {model_path}.")
            if is_adv:
                print(f"Please train on base task '{base_task}' first:")
                print(f"  python agatha_base.py --task {base_task} --train True")
            return

    # Evaluation
    print("\n" + "="*70)
    if is_adv:
        print("AdvGLUE Evaluation (no noise injection)")
    else:
        print("Noisy inference")
    print("="*70)

    noise_levels = [0.0] if is_adv else [0.0, 0.5, 1.0, 2.0, 5.0]

    results = evaluate_with_noise(model, val_loader, eval_task,
                                  noise_levels, args.noise_type, device, args.seed)

    # Save results
    tag_dir = os.path.join(args.model_dir, model_tag)
    os.makedirs(tag_dir, exist_ok=True)
    results_file = os.path.join(tag_dir, f'agatha_base_{eval_task}.json')

    json_results = {}
    for k, metrics in results.items():
        serialized = {}
        for m, v in metrics.items():
            if m == '_iterations_per_layer':
                serialized[m] = {str(layer): iters for layer, iters in v.items()}
            elif isinstance(v, (int, np.integer)):
                serialized[m] = int(v)
            else:
                serialized[m] = float(v)
        json_results[str(k)] = serialized

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "="*70)
    if is_adv:
        print(f"AdvGLUE evaluation results: {eval_task}")
        print(f"(Using model trained on base task: {base_task})")
    print(f"Model: {model_tag}")
    print("="*70)

    # Identify task-specific metrics
    internal_keys = {'avg_iterations', 'convergence_rate'}
    task_metrics = [k for k in results[noise_levels[0]].keys()
                    if k not in internal_keys
                    and not k.startswith('layer')
                    and not k.startswith('_')]

    if is_adv:
        print("\nAdvGLUE task performance:")
    else:
        print(f"\nTask performance vs noise level ({args.noise_type}):")
    print("-" * 110)

    header = f"{'Noise':<10} "
    for metric in task_metrics:
        header += f"{metric.upper():<15} "
        header += f"{metric.upper()+'_DEG%':<15} "
    header += f"{'Avg Iters':<15} {'Conv Rate':<15}"
    print(header)
    print("-" * (10 + 30 * len(task_metrics) + 30))

    baseline_values = {metric: results[noise_levels[0]][metric] for metric in task_metrics}

    for noise_level in noise_levels:
        row = f"{noise_level:<10} "

        for metric in task_metrics:
            metric_value = results[noise_level][metric]
            row += f"{metric_value:<15.4f} "

            if noise_level == 0:
                degradation_str = "baseline"
            else:
                baseline = baseline_values[metric]
                if baseline != 0:
                    degradation = ((baseline - metric_value) / baseline) * 100
                    degradation_str = f"{degradation:+.2f}%"
                else:
                    degradation_str = "N/A"
            row += f"{degradation_str:<15} "

        avg_iters = results[noise_level].get('avg_iterations', 0)
        conv_rate = results[noise_level].get('convergence_rate', 0)
        row += f"{avg_iters:<15.2f} {conv_rate:<15.2%}"
        print(row)

    # Per-layer convergence summary
    num_layers = len(model.hopfield_layers)
    print(f"\n{'='*70}")
    print("Per-layer convergence iterations")
    print("="*70)
    header = f"{'Noise':<10} "
    for layer in range(num_layers):
        name = f"L{layer}"
        header += (
            f"{name + ' Mean':<12} {name + ' Std':<12} {name + ' Med':<12} "
            f"{name + ' Min':<10} {name + ' Max':<10} {name + ' Conv%':<12} "
        )
    print(header)
    print("-" * (10 + 68 * num_layers))

    for noise_level in noise_levels:
        row = f"{noise_level:<10} "
        for layer in range(num_layers):
            prefix = f'layer{layer}'
            m = results[noise_level]
            row += f"{m[f'{prefix}_iterations_mean']:<12.2f} "
            row += f"{m[f'{prefix}_iterations_std']:<12.2f} "
            row += f"{m[f'{prefix}_iterations_median']:<12.1f} "
            row += f"{m[f'{prefix}_iterations_min']:<10d} "
            row += f"{m[f'{prefix}_iterations_max']:<10d} "
            row += f"{m[f'{prefix}_convergence_rate']:<12.2%} "
        print(row)


if __name__ == '__main__':
    main()
