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
This script can be used to evaluate unsupervised models on the IMDb dataset using prompts.
"""

import argparse
import math

import openai
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

from utils import DatasetEntry


class CausalLMWrapper:
    """A wrapper for a causal language model (like GPT-2)"""

    def __init__(self, model_name: str, use_cuda: bool = True):
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_cuda:
            self._model.parallelize()

    def get_token_probabilities(self, input_text: str, prompt: str) -> torch.Tensor:
        input_text = input_text + prompt
        inputs = self._tokenizer.batch_encode_plus([input_text], truncation=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output = self._model(**inputs)['logits']
        return output[:, -1, :]


class MaskedLMWrapper:
    """A wrapper for a masked language model (like BERT)"""

    def __init__(self, model_name: str, use_cuda: bool = True):
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name).to(self._device)

    def get_token_probabilities(self, input_text: str, prompt: str) -> torch.Tensor:
        text_ids = self._tokenizer.encode(input_text, truncation=True, add_special_tokens=False)
        prompt_ids = self._tokenizer.encode(prompt, truncation=False, add_special_tokens=False)

        max_len = self._tokenizer.model_max_length
        max_len_for_text_ids = max_len - len(prompt_ids) - self._tokenizer.num_special_tokens_to_add(False)
        text_ids = text_ids[:max_len_for_text_ids]
        input_ids = text_ids + prompt_ids
        input_ids = torch.tensor([self._tokenizer.build_inputs_with_special_tokens(input_ids)], device=self._device)

        assert sum(1 for id_ in input_ids[0] if id_ == self._tokenizer.mask_token_id) == 1, \
            f"Input text must contain exactly one mask token ('{self._tokenizer.mask_token}'). Got '{input_text}'."
        scores = self._model(input_ids)['logits']
        mask_positions = (input_ids == self._tokenizer.mask_token_id)
        return scores[mask_positions]


class GPT3Wrapper:
    """A wrapper around OpenAI's GPT-3 API"""

    def __init__(self, engine: str):
        self.engine = engine

    def get_scores(self, prompt: str):
        response = openai.Completion.create(engine=self.engine, prompt=prompt, max_tokens=1, logprobs=100)
        top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
        positive_score = max([top_logprobs.get(" good", -math.inf)])
        negative_score = max([top_logprobs.get(" bad", -math.inf)])
        return positive_score, negative_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", choices=["causal", "masked", "gpt3"], required=True,
                        help="The type of the model to evaluate. One of 'causal' (for causal language models like GPT-2), 'masked' (for "
                             "masked language models like BERT), and 'gpt3' (for GPT-3 models accessed via OpenAI's API)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the pretrained model to use (e.g., 'roberta-large')")
    parser.add_argument("--openai_api_key", type=str,
                        help="An optional key for OpenAI's API (only if --model_type is gpt3)")
    parser.add_argument("--test_file", type=str,
                        help="An optional path to a jsonl file of dataset entries. If not given, the entire IMDb dataset is used.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="If set to true, inference is done on CPU only")

    args = parser.parse_args()

    if args.test_file:
        print(f"Evaluating on entries from '{args.test_file}'")
        dataset = DatasetEntry.read_list(args.test_file)
        print(f"Done loading {len(dataset)} examples from '{args.test_file}'")
    else:
        print("Evaluating on the entire IMDb test set")
        dataset = load_dataset('imdb')['test']
        dataset = [DatasetEntry(text_a=text, text_b=None, label=label) for text, label in zip(dataset['text'], dataset['label'])]
        print(f"Done loading {len(dataset)} examples")

    if args.openai_api_key:
        openai.api_key = args.openai_api_key

    predictions, labels = [], []

    if args.model_type == "causal":
        model = CausalLMWrapper(args.model_name, use_cuda=not args.no_cuda)
        prompt = "\nQuestion: Is this movie good or bad?\nAnswer: It is"
    elif args.model_type == "masked":
        model = MaskedLMWrapper(args.model_name, use_cuda=not args.no_cuda)
        prompt = "\nQuestion: Is this movie good or bad?\nAnswer: It is <mask>."
    elif args.model_type == "gpt3":
        model = GPT3Wrapper(args.model_name)
        prompt = "\nQuestion: Is this movie good or bad?\nAnswer: It is"
    else:
        raise ValueError()

    dataset_iterator = tqdm(dataset)
    for ds_entry in dataset_iterator:

        if args.model_type == "gpt3":
            instance_prompt = ds_entry.text_a + prompt
            positive_score, negative_score = model.get_scores(instance_prompt)
        else:
            token_probabilities = model.get_token_probabilities(input_text=ds_entry.text_a, prompt=prompt)[0].detach()
            positive_score = token_probabilities[model._tokenizer.convert_tokens_to_ids("Ġgood")]
            negative_score = token_probabilities[model._tokenizer.convert_tokens_to_ids("Ġbad")]

        labels.append(int(ds_entry.label))
        predictions.append(1 if positive_score > negative_score else 0)

        dataset_iterator.set_description(f"Texts (acc={100 * sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels):5.2f})")
        dataset_iterator.refresh()

    print(f"Final accuracy: {sum(1 for x, y in zip(labels, predictions) if x == y) / len(labels)} (total: {len(labels)})")
