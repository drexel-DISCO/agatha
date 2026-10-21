"""
Unified benchmark dataset loader and metrics.

Loads GLUE, SuperGLUE, AdvGLUE, PAWS / PAWS-X, ANLI and SWAG tasks from the
Hugging Face Hub, tokenizes them for sequence classification and computes the
official task metrics (e.g. Matthews correlation for CoLA, F1 for MRPC / QQP,
F1a and exact match for MultiRC).

Sentence pairs are tokenized as (text_a, text_b) so that the tokenizer produces
the correct token_type_ids for segment embeddings.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from torchmetrics.classification import BinaryMatthewsCorrCoef


# ============================================================================
# Task configurations
# ============================================================================

TASK_CONFIGS = {
    # ========== GLUE Tasks (9 tasks) ==========
    'cola': {
        'num_labels': 2,
        'metric': 'matthews_corrcoef',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'cola',
        'category': 'linguistic_acceptability',
        'benchmark': 'glue'
    },
    'sst2': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'sst2',
        'category': 'sentiment',
        'benchmark': 'glue'
    },
    'mrpc': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'mrpc',
        'category': 'paraphrase',
        'benchmark': 'glue'
    },
    'qqp': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['question1', 'question2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'qqp',
        'category': 'paraphrase',
        'benchmark': 'glue'
    },
    'stsb': {
        'num_labels': 1,
        'metric': 'pearson_spearman',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'stsb',
        'category': 'similarity',
        'benchmark': 'glue'
    },
    'mnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'mnli',
        'category': 'nli',
        'benchmark': 'glue',
        'special_split': 'validation_matched'
    },
    'qnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['question', 'sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'qnli',
        'category': 'qa',
        'benchmark': 'glue'
    },
    'rte': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'rte',
        'category': 'nli',
        'benchmark': 'glue'
    },
    'wnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'wnli',
        'category': 'nli',
        'benchmark': 'glue'
    },

    # ========== AdvGLUE Tasks (Adversarial GLUE - Inference Only) ==========
    'adv_sst2': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_sst2',
        'category': 'sentiment',
        'benchmark': 'advglue',
        'base_task': 'sst2',
        'inference_only': True
    },
    'adv_qqp': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['question1', 'question2'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_qqp',
        'category': 'paraphrase',
        'benchmark': 'advglue',
        'base_task': 'qqp',
        'inference_only': True
    },
    'adv_mnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_mnli',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'mnli',
        'inference_only': True
    },
    'adv_mnli_mismatched': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_mnli_mismatched',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'mnli',
        'inference_only': True
    },
    'adv_qnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['question', 'sentence'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_qnli',
        'category': 'qa',
        'benchmark': 'advglue',
        'base_task': 'qnli',
        'inference_only': True
    },
    'adv_rte': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_rte',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'rte',
        'inference_only': True
    },

    # ========== PAWS (Paraphrase Adversaries from Word Scrambling) ==========
    'paws': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'labeled_final',
        'category': 'paraphrase',
        'benchmark': 'paws'
    },
    'paws_swap': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'labeled_swap',
        'category': 'paraphrase',
        'benchmark': 'paws'
    },
    'paws_unlabeled': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'unlabeled_final',
        'category': 'paraphrase',
        'benchmark': 'paws',
        'noisy_labels': True
    },

    # ========== PAWS-X (Multilingual PAWS) ==========
    'pawsx_en': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'en',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'en'
    },
    'pawsx_de': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'de',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'de'
    },
    'pawsx_es': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'es',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'es'
    },
    'pawsx_fr': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'fr',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'fr'
    },
    'pawsx_zh': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'zh',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'zh'
    },
    'pawsx_ja': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'ja',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'ja'
    },
    'pawsx_ko': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'ko',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'ko'
    },

    # ========== SuperGLUE Tasks (8 tasks) ==========
    'boolq': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['passage', 'question'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'boolq',
        'category': 'reading_comprehension',
        'benchmark': 'superglue'
    },
    'cb': {
        'num_labels': 3,
        'metric': 'accuracy_and_f1',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'cb',
        'category': 'nli',
        'benchmark': 'superglue'
    },
    'copa': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['premise', 'choice1', 'choice2', 'question'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'copa',
        'special_format': True,
        'category': 'reasoning',
        'benchmark': 'superglue',
        'requires_choice_id': True
    },
    'multirc': {
        'num_labels': 2,
        'metric': 'multirc_f1a_em',
        'text_keys': ['paragraph', 'question', 'answer'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'multirc',
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'superglue',
        'requires_question_id': True
    },
    'record': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['passage', 'query', 'entities'],
        'label_key': 'answers',
        'dataset_name': 'super_glue',
        'task_name': 'record',
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'superglue',
        'note': 'Using entity ranking with binary F1 (simplified from official token F1+EM)'
    },
    'wic': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2', 'word'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'wic',
        'category': 'word_sense',
        'benchmark': 'superglue'
    },
    'wsc': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['text', 'span1_text', 'span2_text'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'wsc',
        'special_format': True,
        'category': 'reasoning',
        'benchmark': 'superglue'
    },
    'axb': {
        'num_labels': 2,
        'metric': 'matthews_corrcoef',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'axb',
        'category': 'diagnostic',
        'benchmark': 'superglue',
        'test_only': True
    },
    'axg': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'axg',
        'category': 'diagnostic',
        'benchmark': 'superglue',
        'test_only': True
    },

    # ========== ANLI (Adversarial Natural Language Inference) ==========
    'anli_r1': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,
        'category': 'nli',
        'benchmark': 'anli',
        'round': 1,
        'special_splits': True
    },
    'anli_r2': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,
        'category': 'nli',
        'benchmark': 'anli',
        'round': 2,
        'special_splits': True
    },
    'anli_r3': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,
        'category': 'nli',
        'benchmark': 'anli',
        'round': 3,
        'special_splits': True
    },

    # ========== SWAG (Situations With Adversarial Generations) ==========
    'swag': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3'],
        'label_key': 'label',
        'dataset_name': 'swag',
        'task_name': 'regular',
        'special_format': True,
        'category': 'commonsense_reasoning',
        'benchmark': 'swag',
        'requires_choice_id': True
    },
}


# ============================================================================
# Dataset
# ============================================================================

class BenchmarkDataset(Dataset):
    """
    PyTorch Dataset over tokenized inputs, with optional per-example metadata
    (e.g. question IDs for MultiRC).
    """

    def __init__(self, encodings, labels, metadata=None):
        """
        Args:
            encodings: Tokenized inputs (contains input_ids, attention_mask, token_type_ids)
            labels: Target labels
            metadata: Optional dict with additional info (e.g., {'question_id': [...]})
        """
        self.encodings = encodings
        self.labels = labels
        self.metadata = metadata or {}

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if key != 'label':
                item[key] = torch.tensor(val[idx])
        item['label'] = torch.tensor(self.labels[idx])

        # Add metadata if available
        for key, val in self.metadata.items():
            item[key] = val[idx]

        return item

    def __len__(self):
        return len(self.labels)


# ============================================================================
# Text preprocessing functions
#
# All functions return tuples with (text_a, text_b, label, ...optional_metadata).
# text_b is None for single-sentence tasks.
# The tokenizer is called as tokenizer(texts_a, texts_b) for pair tasks,
# which produces correct token_type_ids (0 for sentence A, 1 for sentence B).
# ============================================================================

def preprocess_glue_single(example: Dict, text_key: str) -> Tuple[str, None, int]:
    """Single-sentence task. Returns (text_a, None, label)."""
    text = str(example[text_key])
    label = int(example['label'])
    return text, None, label


def preprocess_glue_pair(example: Dict, text_key1: str, text_key2: str) -> Tuple[str, str, int]:
    """Sentence-pair task. Returns (text_a, text_b, label)."""
    text_a = str(example[text_key1])
    text_b = str(example[text_key2])
    label = int(example['label'])
    return text_a, text_b, label


def preprocess_stsb(example: Dict) -> Tuple[str, str, float]:
    """STS-B regression task. Returns (text_a, text_b, label)."""
    text_a = str(example['sentence1'])
    text_b = str(example['sentence2'])
    label = float(example['label'])
    return text_a, text_b, label


def preprocess_multirc(example: Dict) -> Tuple[str, str, int, str]:
    """
    MultiRC: paragraph+question as text_a, answer as text_b.
    Returns (text_a, text_b, label, question_id).
    """
    text_a = f"{example['paragraph']} Question: {example['question']}"
    text_b = f"Answer: {example['answer']}"
    label = int(example['label'])

    if 'idx' in example and 'question' in example['idx']:
        question_id = f"{example['idx']['paragraph']}_{example['idx']['question']}"
    else:
        question_id = str(hash(f"{example['paragraph']}|||{example['question']}"))

    return text_a, text_b, label, question_id


def preprocess_record(example: Dict) -> List[Tuple[str, str, int]]:
    """
    ReCoRD: passage as text_a, filled query as text_b.
    Returns list of (text_a, text_b, label) tuples - one per candidate entity.
    """
    passage = str(example['passage'])
    query = example['query']
    entities = example['entities']
    answers = example['answers']

    examples = []
    for entity in entities:
        filled_query = query.replace('@placeholder', entity)
        label = 1 if entity in answers else 0
        examples.append((passage, filled_query, label))

    return examples


def preprocess_copa(example: Dict) -> List[Tuple[str, str, int, str, int]]:
    """
    COPA: premise+question as text_a, choice as text_b.
    Returns list of (text_a, text_b, label, copa_id, choice_num) tuples.
    """
    premise = example['premise']
    choice1 = example['choice1']
    choice2 = example['choice2']
    question = example['question']
    correct_choice = int(example['label'])

    copa_id = str(example.get('idx', hash(premise)))

    text_a = f"{premise} What was the {question}?"

    examples = []
    examples.append((text_a, choice1, 1 if correct_choice == 0 else 0, copa_id, 0))
    examples.append((text_a, choice2, 1 if correct_choice == 1 else 0, copa_id, 1))

    return examples


def preprocess_swag(example: Dict) -> List[Tuple[str, str, int, str, int]]:
    """
    SWAG: context as text_a, ending as text_b.
    Returns list of (text_a, text_b, label, swag_id, choice_num) tuples.
    """
    sent1 = example['sent1']
    sent2 = example.get('sent2', '')

    if sent2:
        context = f"{sent1} {sent2}"
    else:
        context = sent1

    endings = [
        example['ending0'],
        example['ending1'],
        example['ending2'],
        example['ending3']
    ]

    correct_choice = int(example['label'])

    swag_id = str(example.get('video-id', '') + '_' + example.get('fold-ind', ''))
    if not swag_id.strip('_'):
        swag_id = str(hash(context))

    examples = []
    for choice_num, ending in enumerate(endings):
        label = 1 if choice_num == correct_choice else 0
        examples.append((context, ending, label, swag_id, choice_num))

    return examples


def preprocess_wsc(example: Dict) -> Tuple[str, str, int]:
    """
    WSC: text as text_a, coreference question as text_b.
    Returns (text_a, text_b, label).
    """
    text_str = str(example['text'])
    span1 = example['span1_text']
    span2 = example['span2_text']

    text_b = f"Does '{span2}' refer to '{span1}'?"
    label = int(example['label'])

    return text_str, text_b, label


def preprocess_wic(example: Dict) -> Tuple[str, str, int]:
    """
    WiC: first sentence+word as text_a, second sentence as text_b.
    Returns (text_a, text_b, label).
    """
    sentence1 = str(example['sentence1'])
    sentence2 = str(example['sentence2'])
    word = example['word']

    text_a = f"{sentence1} Word: {word}"
    label = int(example['label'])

    return text_a, sentence2, label


# ============================================================================
# Preprocess function mapping
# ============================================================================

PREPROCESS_FUNCTIONS = {
    # GLUE
    'cola': lambda x: preprocess_glue_single(x, 'sentence'),
    'sst2': lambda x: preprocess_glue_single(x, 'sentence'),
    'mrpc': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'qqp': lambda x: preprocess_glue_pair(x, 'question1', 'question2'),
    'stsb': preprocess_stsb,
    'mnli': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'qnli': lambda x: preprocess_glue_pair(x, 'question', 'sentence'),
    'rte': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'wnli': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # SuperGLUE
    'boolq': lambda x: preprocess_glue_pair(x, 'passage', 'question'),
    'cb': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'copa': preprocess_copa,
    'swag': preprocess_swag,
    'wic': preprocess_wic,
    'wsc': preprocess_wsc,
    'multirc': preprocess_multirc,
    'record': preprocess_record,
    # ANLI
    'anli_r1': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'anli_r2': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'anli_r3': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    # Diagnostics
    'axb': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'axg': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    # AdvGLUE
    'adv_sst2': lambda x: preprocess_glue_single(x, 'sentence'),
    'adv_qqp': lambda x: preprocess_glue_pair(x, 'question1', 'question2'),
    'adv_mnli': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'adv_mnli_mismatched': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'adv_qnli': lambda x: preprocess_glue_pair(x, 'question', 'sentence'),
    'adv_rte': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # PAWS
    'paws': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'paws_swap': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'paws_unlabeled': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # PAWS-X
    'pawsx_en': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_de': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_es': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_fr': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_zh': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_ja': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_ko': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
}


# ============================================================================
# Internal helpers
# ============================================================================

def _process_split(data, task_name, preprocess_fn, requires_question_id, requires_choice_id):
    """
    Process a dataset split into parallel lists.

    Returns:
        texts_a: list of str (first sentences)
        texts_b: list of str or None (second sentences, None for single-sentence tasks)
        labels: list of labels
        metadata: dict with optional question_id, copa_id, choice_num lists
    """
    texts_a = []
    texts_b = []
    labels = []
    question_ids = [] if requires_question_id else None
    copa_ids = [] if requires_choice_id else None
    choice_nums = [] if requires_choice_id else None

    for example in data:
        try:
            result = preprocess_fn(example)

            if task_name == 'record':
                # Returns list of (text_a, text_b, label)
                for text_a, text_b, label in result:
                    texts_a.append(text_a)
                    texts_b.append(text_b)
                    labels.append(label)

            elif task_name == 'copa':
                # Returns list of (text_a, text_b, label, copa_id, choice_num)
                for text_a, text_b, label, cid, cnum in result:
                    texts_a.append(text_a)
                    texts_b.append(text_b)
                    labels.append(label)
                    copa_ids.append(cid)
                    choice_nums.append(cnum)

            elif task_name == 'swag':
                # Returns list of (text_a, text_b, label, swag_id, choice_num)
                for text_a, text_b, label, sid, cnum in result:
                    texts_a.append(text_a)
                    texts_b.append(text_b)
                    labels.append(label)
                    copa_ids.append(sid)
                    choice_nums.append(cnum)

            elif requires_question_id:
                # multirc: returns (text_a, text_b, label, question_id)
                text_a, text_b, label, qid = result
                texts_a.append(text_a)
                texts_b.append(text_b)
                labels.append(label)
                question_ids.append(qid)

            else:
                # Standard: returns (text_a, text_b, label) where text_b may be None
                text_a, text_b, label = result
                texts_a.append(text_a)
                texts_b.append(text_b)
                labels.append(label)

        except Exception as e:
            print(f"Warning: Skipping example: {e}")
            continue

    # Build metadata dict
    metadata = {}
    if requires_question_id:
        metadata['question_id'] = question_ids
    if requires_choice_id:
        metadata['copa_id'] = copa_ids
        metadata['choice_num'] = choice_nums

    return texts_a, texts_b, labels, metadata


def _tokenize(tokenizer, texts_a, texts_b, max_length):
    """
    Tokenize with proper pair handling.

    If texts_b contains non-None values, tokenizes as sentence pairs so
    token_type_ids correctly marks segment A (0) vs segment B (1).
    """
    is_pair = any(t is not None for t in texts_b)

    if is_pair:
        # Replace any remaining None values with empty string (shouldn't happen
        # in practice, but guards against mixed single/pair in same split)
        texts_b_clean = [t if t is not None else '' for t in texts_b]
        encodings = tokenizer(
            texts_a,
            texts_b_clean,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
    else:
        encodings = tokenizer(
            texts_a,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )

    return encodings


# ============================================================================
# Data loading
# ============================================================================

def load_data(task_name: str, tokenizer, max_length: int = 128, max_train_samples: Optional[int] = None) -> Tuple[BenchmarkDataset, BenchmarkDataset]:
    """
    Load and preprocess any benchmark dataset.

    Sentence pairs are tokenized as (text_a, text_b) so the tokenizer produces
    correct token_type_ids for segment embeddings.

    Returns (train_dataset, eval_dataset). The evaluation set is the validation
    split (validation_matched for MNLI); train_dataset is None for
    inference-only (AdvGLUE) and test-only (diagnostic) tasks.
    """

    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")

    task_config = TASK_CONFIGS[task_name]
    requires_question_id = task_config.get('requires_question_id', False)
    requires_choice_id = task_config.get('requires_choice_id', False)
    is_test_only = task_config.get('test_only', False)
    is_inference_only = task_config.get('inference_only', False)

    print(f"Loading task: {task_name} ({task_config['benchmark'].upper()})")

    # Load dataset
    dataset_name = task_config['dataset_name']
    task_name_param = task_config.get('task_name')

    if task_name_param:
        dataset = load_dataset(dataset_name, task_name_param)
    else:
        dataset = load_dataset(dataset_name)

    preprocess_fn = PREPROCESS_FUNCTIONS[task_name]

    # ------------------------------------------------------------------
    # Test-only datasets (axb, axg)
    # ------------------------------------------------------------------
    if is_test_only:
        print("Processing test data (diagnostic task)...")
        texts_a, texts_b, test_labels, test_metadata = _process_split(
            dataset['test'], task_name, preprocess_fn, requires_question_id, requires_choice_id
        )

        test_encodings = _tokenize(tokenizer, texts_a, texts_b, max_length)
        test_dataset = BenchmarkDataset(test_encodings, test_labels, test_metadata if test_metadata else None)

        print(f"[+] Loaded {len(test_dataset)} test examples (no training data)")
        return None, test_dataset

    # ------------------------------------------------------------------
    # Inference-only datasets (AdvGLUE)
    # ------------------------------------------------------------------
    if is_inference_only:
        print("Processing inference-only dataset (no training data)...")
        val_split = 'validation' if 'validation' in dataset else 'dev'

        texts_a, texts_b, val_labels, val_metadata = _process_split(
            dataset[val_split], task_name, preprocess_fn, requires_question_id, requires_choice_id
        )

        val_encodings = _tokenize(tokenizer, texts_a, texts_b, max_length)
        val_dataset = BenchmarkDataset(val_encodings, val_labels, val_metadata if val_metadata else None)

        print(f"[+] Loaded {len(val_dataset)} validation examples (inference-only, no training data)")
        return None, val_dataset

    # ------------------------------------------------------------------
    # Standard train + validation
    # ------------------------------------------------------------------

    # Determine splits
    if task_name.startswith('anli_'):
        round_num = task_config.get('round')
        train_split = f'train_r{round_num}'
        val_split = f'dev_r{round_num}'
        print(f"  Using ANLI Round {round_num} splits: {train_split}, {val_split}")
    else:
        val_split = task_config.get('special_split', 'validation')
        train_split = 'train'

    # Process training data
    print("Processing training data...")
    train_data = dataset[train_split]
    if max_train_samples is not None and max_train_samples < len(train_data):
        print(f"  Using subset: {max_train_samples} of {len(train_data)} samples")
        train_data = train_data.select(range(max_train_samples))

    train_texts_a, train_texts_b, train_labels, train_metadata = _process_split(
        train_data, task_name, preprocess_fn, requires_question_id, requires_choice_id
    )

    # Process validation data
    print("Processing validation data...")
    val_texts_a, val_texts_b, val_labels, val_metadata = _process_split(
        dataset[val_split], task_name, preprocess_fn, requires_question_id, requires_choice_id
    )

    # Tokenize
    print("Tokenizing...")
    train_encodings = _tokenize(tokenizer, train_texts_a, train_texts_b, max_length)
    val_encodings = _tokenize(tokenizer, val_texts_a, val_texts_b, max_length)

    # Create datasets
    train_dataset = BenchmarkDataset(train_encodings, train_labels, train_metadata if train_metadata else None)
    val_dataset = BenchmarkDataset(val_encodings, val_labels, val_metadata if val_metadata else None)

    print(f"[+] Loaded {len(train_dataset)} training examples")
    print(f"[+] Loaded {len(val_dataset)} validation examples")
    if requires_question_id and val_metadata.get('question_id'):
        num_unique_questions = len(set(val_metadata['question_id']))
        print(f"[+] Unique questions in validation: {num_unique_questions}")
    if requires_choice_id and val_metadata.get('copa_id'):
        num_unique_copa = len(set(val_metadata['copa_id']))
        print(f"[+] Unique choice questions in validation: {num_unique_copa}")

    return train_dataset, val_dataset


# ============================================================================
# Metrics
# ============================================================================

def compute_multirc_metrics(predictions, labels, question_ids):
    """
    Compute F1a and EM for MultiRC.
    F1a: average of per-question F1 scores.
    EM: Exact Match per question (1 if all answers for a question are correct, 0 otherwise).
    """
    question_groups = defaultdict(lambda: {'preds': [], 'labels': []})
    for pred, label, qid in zip(predictions, labels, question_ids):
        question_groups[qid]['preds'].append(pred)
        question_groups[qid]['labels'].append(label)

    f1_scores = []
    em_scores = []

    for qid, data in question_groups.items():
        # Compute F1 for this specific question
        f1 = f1_score(data['labels'], data['preds'], average='binary', zero_division=0)
        f1_scores.append(f1)

        # Compute Exact Match for this specific question
        # It's an exact match ONLY if the arrays match completely
        is_exact_match = int(data['labels'] == data['preds'])
        em_scores.append(is_exact_match)

    f1a = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    em = sum(em_scores) / len(em_scores) if em_scores else 0.0

    return {
        'f1a': f1a,
        'em': em,
        'avg': (f1a + em) / 2.0
    }


def compute_metrics(task_name: str, predictions, labels, metadata=None, include_accuracy: bool = False) -> Dict[str, float]:
    """
    Compute metrics for any benchmark task.

    The first entry of the returned dict is the task's primary metric, which
    is used for model selection. With include_accuracy=True, accuracy is
    added for tasks whose primary metric is not accuracy.
    """

    task_config = TASK_CONFIGS[task_name]
    metric_type = task_config['metric']

    if metric_type == 'accuracy':
        # Special handling for COPA
        if task_name == 'copa' and metadata and 'copa_id' in metadata:
            copa_groups = defaultdict(lambda: {'preds': [], 'labels': [], 'choice_nums': []})
            for pred, label, copa_id, choice_num in zip(predictions, labels,
                                                         metadata['copa_id'], metadata['choice_num']):
                copa_groups[copa_id]['preds'].append(pred)
                copa_groups[copa_id]['labels'].append(label)
                copa_groups[copa_id]['choice_nums'].append(choice_num)

            correct = 0
            total = 0
            for copa_id, data in copa_groups.items():
                if data['preds'][0] > data['preds'][1]:
                    predicted_choice = 0
                elif data['preds'][1] > data['preds'][0]:
                    predicted_choice = 1
                else:
                    predicted_choice = 0

                correct_choice = 0 if data['labels'][0] == 1 else 1
                if predicted_choice == correct_choice:
                    correct += 1
                total += 1

            return {'accuracy': correct / total if total > 0 else 0.0}

        # Special handling for SWAG
        elif task_name == 'swag' and metadata and 'copa_id' in metadata:
            swag_groups = defaultdict(lambda: {'preds': [], 'labels': [], 'choice_nums': []})
            for pred, label, swag_id, choice_num in zip(predictions, labels,
                                                         metadata['copa_id'], metadata['choice_num']):
                swag_groups[swag_id]['preds'].append(pred)
                swag_groups[swag_id]['labels'].append(label)
                swag_groups[swag_id]['choice_nums'].append(choice_num)

            correct = 0
            total = 0
            for swag_id, data in swag_groups.items():
                predicted_choice = max(range(len(data['preds'])), key=lambda i: data['preds'][i])
                correct_choice = data['labels'].index(1) if 1 in data['labels'] else 0

                if predicted_choice == correct_choice:
                    correct += 1
                total += 1

            return {'accuracy': correct / total if total > 0 else 0.0}

        else:
            return {'accuracy': accuracy_score(labels, predictions)}

    elif metric_type == 'accuracy_and_f1':
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {
            'accuracy': acc,
            'f1': f1,
            'avg': (acc + f1) / 2
        }

    elif metric_type == 'f1':
        num_labels = task_config['num_labels']
        average = 'binary' if num_labels == 2 else 'macro'
        result = {'f1': f1_score(labels, predictions, average=average)}
        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)
        return result

    elif metric_type == 'multirc_f1a_em':
        if metadata is None or 'question_id' not in metadata:
            raise ValueError("MultiRC metric requires question_id in metadata")

        question_ids = metadata['question_id']
        result = compute_multirc_metrics(predictions, labels, question_ids)

        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)

        return result

    elif metric_type == 'matthews_corrcoef':
        metric = BinaryMatthewsCorrCoef()
        mcc = metric(torch.tensor(predictions), torch.tensor(labels))
        result = {'matthews_corrcoef': mcc.item()}
        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)
        return result

    elif metric_type == 'pearson_spearman':
        # Cast to Python floats so that checkpoints storing these metrics can
        # be reloaded with torch.load(weights_only=True).
        pearson = float(pearsonr(predictions, labels)[0])
        spearman = float(spearmanr(predictions, labels)[0])
        result = {'spearman': spearman}
        if include_accuracy:
            result['pearson'] = pearson
            result['pearson_spearman'] = (pearson + spearman) / 2
        return result

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# ============================================================================
# Utility functions
# ============================================================================

def get_task_info(task_name: str) -> Dict:
    """Get configuration for a task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_CONFIGS[task_name]


if __name__ == '__main__':
    tasks_by_benchmark = defaultdict(list)
    for name, config in TASK_CONFIGS.items():
        tasks_by_benchmark[config['benchmark']].append(name)

    print("Supported tasks:")
    for benchmark, names in tasks_by_benchmark.items():
        print(f"  {benchmark:<10} {', '.join(names)}")
