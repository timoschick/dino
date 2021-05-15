# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate a regular supervised model trained with DINO datasets on the IMDb dataset.
"""

import argparse
import os
from typing import Dict

import torch
import torch.utils.data

from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EvaluationStrategy

from utils import DatasetEntry


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_datasets(args) -> Dict[str, IMDbDataset]:
    dino_ds = DatasetEntry.read_list(args.input_file)
    imdb_ds = load_dataset("")['test']
    imdb500_ds = DatasetEntry.read_list(args.imdb_500_file)

    train_texts, train_labels = [x.text_b for x in dino_ds], [int(x.label) for x in dino_ds]
    test_texts, test_labels = [x.replace('<br />', '\n') for x in imdb_ds['text']], imdb_ds['label']
    imdb500_texts, imdb500_labels = [x.text_a for x in imdb500_ds], [int(x.label) for x in imdb500_ds]

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.1, random_state=42)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    imdb500_encodings = tokenizer(imdb500_texts, truncation=True, padding=True)

    return {
        'train': IMDbDataset(train_encodings, train_labels),
        'val': IMDbDataset(val_encodings, val_labels),
        'test': IMDbDataset(test_encodings, test_labels),
        'imdb500': IMDbDataset(imdb500_encodings, imdb500_labels)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to an output directory were the finetuned model and results are saved")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to a jsonl file generated with DINO containing the training dataset")
    parser.add_argument("--imdb_500_file", type=str, required=True,
                        help="Path to the IMDb-500 dataset for evaluation")

    parser.add_argument("--model", type=str, default="roberta-base",
                        help="Name of the pretrained model to use for finetuning")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="The training batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                        help="The eval batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="The number of gradient accumulation steps to perform")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="The maximum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="The number of initial warmup steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="The maximum number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="The maximum number of training epochs")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 100, 123],
                        help="The seeds to use. If multiple are given, the entire finetuning process is repeated multiple times.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    metric = load_metric("accuracy")

    datasets = load_datasets(args)

    for seed in args.seeds:
        output_dir = os.path.join(args.output_dir, str(seed))

        training_args = TrainingArguments(
            output_dir=output_dir, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            weight_decay=0.01, logging_dir='./logs', logging_steps=100, evaluation_strategy=EvaluationStrategy.STEPS,
            load_best_model_at_end=True, metric_for_best_model="accuracy", seed=seed
        )

        trainer = Trainer(
            model=model, args=training_args, train_dataset=datasets['train'], eval_dataset=datasets['val'], compute_metrics=compute_metrics
        )

        trainer.train()

        with open(os.path.join(args.output_dir, f'results-{seed}.txt'), 'w', encoding='utf8') as fh:
            print("Evaluating on IMDb500")
            result_imdb500 = trainer.evaluate(eval_dataset=datasets['imdb500'])
            print(result_imdb500)
            fh.write("=== IMDb500 ===\n")
            fh.write(str(result_imdb500) + '\n\n')

            print("Evaluating on IMDb test")
            result_test = trainer.evaluate(eval_dataset=datasets['test'])
            print(result_test)
            fh.write("=== IMDb test ===\n")
            fh.write(str(result_test) + '\n')
