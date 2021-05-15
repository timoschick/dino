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
This module contains some utility functions.
"""

import csv
import json
import random
from typing import List, Optional, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_inputs(input_file: str, input_file_type: str) -> List[str]:
    """
    Read a list of input texts from a text file.
    :param input_file: the path to the input file
    :param input_file_type: the file type, one of 'plain', 'jsonl' and 'stsb':
        <ul>
            <li>'plain': a plain text file where each line corresponds to one input</li>
            <li>'jsonl': a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'</li>
            <li>'stsb': a tsv file, formatted like the official STS benchmark</li>
        </ul>
    :return: the list of extracted input texts
    """
    valid_types = ['plain', 'jsonl', 'stsb']
    assert input_file_type in valid_types, f"Invalid input file type: '{input_file_type}'. Valid types: {valid_types}"

    if input_file_type == "plain":
        return read_plaintext_inputs(input_file)
    elif input_file_type == "jsonl":
        return read_jsonl_inputs(input_file)
    elif input_file_type == "stsb":
        return read_sts_inputs(input_file)


def read_plaintext_inputs(path: str) -> List[str]:
    """Read input texts from a plain text file where each line corresponds to one input"""
    with open(path, 'r', encoding='utf8') as fh:
        inputs = fh.read().splitlines()
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs


def read_jsonl_inputs(path: str) -> List[str]:
    """Read input texts from a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'"""
    ds_entries = DatasetEntry.read_list(path)
    print(f"Done loading {len(ds_entries)} inputs from file '{path}'")
    return [entry.text_a for entry in ds_entries]


def read_sts_inputs(path: str) -> List[str]:
    """Read input texts from a tsv file, formatted like the official STS benchmark"""
    inputs = []
    with open(path, 'r', encoding='utf8') as fh:
        reader = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                sent_a, sent_b = row[5], row[6]
                inputs.append(sent_a)
                inputs.append(sent_b)
            except IndexError:
                print(f"Cannot parse line {row}")
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs


class DatasetEntry:
    """This class represents a dataset entry for text (pair) classification"""

    def __init__(self, text_a: str, text_b: Optional[str], label: Any):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        if self.text_b is not None:
            return f'DatasetEntry(text_a="{self.text_a}", text_b="{self.text_b}", label={self.label})'
        else:
            return f'DatasetEntry(text_a="{self.text_a}", label={self.label})'

    def __key(self):
        return self.text_a, self.text_b, self.label

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, DatasetEntry):
            return self.__key() == other.__key()
        return False

    @staticmethod
    def save_list(entries: List['DatasetEntry'], path: str):
        with open(path, 'w', encoding='utf8') as fh:
            for entry in entries:
                fh.write(f'{json.dumps(entry.__dict__)}\n')

    @staticmethod
    def read_list(path: str) -> List['DatasetEntry']:
        pairs = []
        with open(path, 'r', encoding='utf8') as fh:
            for line in fh:
                pairs.append(DatasetEntry(**json.loads(line)))
        return pairs
