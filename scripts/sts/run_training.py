import argparse
import logging
import random
import os
import gzip
import csv
import math
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models, util, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from utils import DatasetEntry


def download_sts_dataset(sts_dataset_path: str) -> None:
    """Download the STS dataset if it isn't already present."""
    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Build the Sentence Transformer model."""
    try:
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    except OSError:
        return SentenceTransformer(model_name)


def split_dataset(ds: List[DatasetEntry], dev_size: float = 0.1, seed: int = 42) -> Dict[str, List[DatasetEntry]]:
    """Split a dataset into a train and dev set.

    The split is performed such that the distribution of labels is identical for the training and development set.

    :param ds: The dataset to split.
    :param dev_size: The relative size of the development set, in the range (0,1).
    :param seed: The seed used to initialize the random number generator.
    :return: A dictionary with keys "train" and "dev", whose values are the corresponding datasets.
    """
    train, dev = [], []
    rng = random.Random(seed)
    ds_grouped_by_label = defaultdict(list)
    for x in ds:
        ds_grouped_by_label[x.label].append(x)

    for label_list in ds_grouped_by_label.values():
        rng.shuffle(label_list)
        num_dev_examples = int(len(label_list) * dev_size)
        train += label_list[num_dev_examples:]
        dev += label_list[:num_dev_examples]

    return {'train': train, 'dev': dev}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", type=str, required=True,
                        help="The JSONL file that contains the DINO-generated dataset to train on.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory for storing the trained model and evaluation results.")

    # Model and training parameters
    parser.add_argument("--model_name", type=str, default='roberta-base',
                        help="The pretrained Transformer language model to use.")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="The batch size used for training.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="The number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used to initialize all random number generators.")

    # Evaluation parameters
    parser.add_argument("--sts_dataset_path", type=str, default="datasets/stsbenchmark.tsv.gz",
                        help="The path to the STSb dataset. The STSb dataset is downloaded and saved at this path if it does not exist.")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    input_filename = os.path.basename(args.input_file)

    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    # We write all arguments to a file for better reproducibility.
    args_file = os.path.join(args.output_dir, f'args-{input_filename}.jsonl')
    with open(args_file, 'w', encoding='utf8') as fh:
        fh.write(str(vars(args)))

    # If the STSb dataset does not exist, we download it.
    download_sts_dataset(args.sts_dataset_path)

    model = build_sentence_transformer(args.model_name)
    model_save_name = '_'.join([input_filename, args.model_name.replace("/", "-"), args.date.replace("/", "-").replace(" ", "_")])
    model_save_path = os.path.join(args.output_dir, model_save_name)

    # Load and split the (postprocessed) STS-DINO dataset.
    dataset = DatasetEntry.read_list(args.input_file)
    dataset = split_dataset(dataset, dev_size=0.1, seed=args.seed)

    train_samples = [InputExample(texts=[x.text_a, x.text_b], label=x.label) for x in dataset['train']]
    dev_samples = [InputExample(texts=[x.text_a, x.text_b], label=x.label) for x in dataset['dev']]

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dino-dev')

    # We use 10% of the training data for warm-up.
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)

    # Train the model.
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=args.num_epochs,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    # Load the trained model and perform evaluation.
    if args.num_epochs > 0:
        model = SentenceTransformer(model_save_path)

    results = OrderedDict()
    stsb_samples = []

    with gzip.open(args.sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1.
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'test':
                stsb_samples.append(inp_example)

    stsb_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(stsb_samples, name='stsb-test')
    results['stsb'] = stsb_evaluator(model, output_path='.')

    print(results)
    with open(os.path.join(args.output_dir, f'{input_filename}-results.txt'), 'w', encoding='utf8') as fh:
        for task, result in results.items():
            fh.write(f'{task}: {result}\n')
