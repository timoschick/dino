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
This script can be used to generate datasets with DINO (Datasets from Instructions).
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict

from modeling import GPT2Wrapper, DinoGenerator, PLACEHOLDER_STR
from utils import set_seed, read_inputs, DatasetEntry


def validate_args(args) -> None:
    """Validate the given command line arguments"""
    if args.input_file is not None:
        assert args.num_entries_per_input_and_label is not None, "If 'input_file' is set, 'num_entries_per_input_and_label' must be set"
        assert args.num_entries_per_label is None, "If 'input_file' is set, 'num_entries_per_label' must not be set"
        assert args.batch_size is None, "If 'input_file' is set, batch_size must not be set as 'num_entries_per_input_and_label' also " \
                                        "serves as batch size in this case"
    else:
        assert args.num_entries_per_input_and_label is None, "If 'input_file' is not set, 'num_entries_per_input_and_label' must not be set"
        assert args.num_entries_per_label is not None, "If 'input_file' is not set, 'num_entries_per_label' must be set"
        assert args.batch_size is not None, "If 'input_file' is not set, 'batch_size' must be set"


def validate_task_spec(task_spec: Dict[str, Any], with_inputs: bool) -> None:
    """Validate the given task specification"""
    error_prefix = "Invalid task specification:"
    assert 'task_name' in task_spec, f"{error_prefix} missing field 'task_name'"
    assert isinstance(task_spec['task_name'], str) and re.match(r"^[A-Za-z0-9\-_.]+$", task_spec['task_name']), \
        f"{error_prefix} 'task_name' must be a string consisting only of [A-Za-z0-9\\-_.]"
    assert 'labels' in task_spec, f"{error_prefix} missing field 'labels'"
    assert isinstance(task_spec['labels'], dict), f"{error_prefix} 'labels' must be a dictionary"
    all_labels = task_spec['labels'].keys()
    for label, label_dict in task_spec['labels'].items():
        assert isinstance(label_dict, dict), f"{error_prefix} label '{label}' is not mapped to a dictionary"
        assert not label_dict.keys() - {'instruction', 'counter_labels'}, \
            f"{error_prefix} invalid keys for label '{label}', only 'instruction' and 'counter_labels' are allowed"
        assert 'instruction' in label_dict.keys(), f"{error_prefix} missing field 'instruction' for label '{label}'"
        assert isinstance(label_dict['instruction'], str), f"{error_prefix} 'instruction' not a string for label '{label}'"
        assert label_dict['instruction'][-1] == '"', \
            f"{error_prefix} each instruction should end with an opening quotation mark (\") so that the next quotation mark generated " \
            f"by the model can be interpreted as a signal that it is done."
        if with_inputs:
            assert label_dict['instruction'].count(PLACEHOLDER_STR) == 1, \
                f"{error_prefix} The instruction for label '{label}' does not contain exactly one placeholder token ({PLACEHOLDER_STR}). " \
                f"If an input file is specified, each instruction must contain this placeholder to indicate where the input should be " \
                f"inserted."
        else:
            assert label_dict['instruction'].count(PLACEHOLDER_STR) == 0, \
                f"{error_prefix} The instruction for label '{label}' contains a placeholder token ({PLACEHOLDER_STR}). If no input file " \
                f"is specified, instructions must not contain this placeholder as there is no input to replace it with."
        if 'counter_labels' in label_dict.keys():
            assert isinstance(label_dict['counter_labels'], list), f"{error_prefix} 'counter_labels' not a list for label '{label}'"
            for counter_label in label_dict['counter_labels']:
                assert counter_label in all_labels, f"{error_prefix} counter_label '{counter_label}' for label '{label}' is not a label"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--task_file", type=str, required=True,
                        help="A json file providing the instructions and other information required for dataset generation. "
                             "See the 'task_specs' directory for examples and 'README.md' for more details on how to create this file.")

    # Text generation and sampling parameters
    parser.add_argument("--model_name", type=str, default="gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--max_output_length", type=int, default=40,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--decay_constant", type=float, default=100,
                        help="The decay constant for self-debiasing")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")

    # Dataset parameters
    parser.add_argument("--input_file", type=str,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")
    parser.add_argument("--input_file_type", choices=["plain", "jsonl", "stsb"], default="plain",
                        help="The type of the input file. Choices are 'plain' (a raw text file with one input per line), 'jsonl' (a jsonl "
                             "file as produced by DINO) and 'stsb' (a TSV file in the STS Benchmark format)")
    parser.add_argument("--num_entries_per_input_and_label", type=int, default=None,
                        help="The number of entries to generate for each pair of input text and label (only if --input_file is set)")
    parser.add_argument("--num_entries_per_label", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--remove_duplicates", action='store_true',
                        help="Whether duplicates should be removed from the generated dataset")
    parser.add_argument("--remove_identical_pairs", action='store_true',
                        help="Whether text pairs with text_a == text_b should be removed from the dataset (only for text pair datasets)")
    parser.add_argument("--keep_outputs_without_eos", action='store_true',
                        help="If set to true, examples where the language model does not output a quotation mark (which is interpreted as "
                             "a signal that it has completed its output) are not removed from the dataset.")
    parser.add_argument("--allow_newlines_in_outputs", action='store_true',
                        help="If set to true, model outputs that contain a newline character before the end-of-sequence token (a quotation "
                             "mark) are not removed from the dataset.")
    parser.add_argument("--min_num_words", type=int, default=-1,
                        help="The minimum number of (whitespace-separated) words for each dataset entry. Entries with fewer words are "
                             "removed.")
    parser.add_argument("--min_num_tokens", type=int, default=-1,
                        help="The minimum number of tokens for each dataset entry. Entries with fewer tokens are removed.")

    # Miscellaneous further parameters
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    validate_args(args)

    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.task_file, 'r', encoding='utf8') as fh:
        task_specification = json.load(fh)
        validate_task_spec(task_specification, with_inputs=args.input_file is not None)

    args_file = os.path.join(args.output_dir, f'{task_specification["task_name"]}-args.json')
    with open(args_file, 'w', encoding='utf8') as fh:
        fh.write(json.dumps(vars(args), indent=4))

    inputs = read_inputs(args.input_file, args.input_file_type) if args.input_file else None

    if args.openai_api_key:
        print(f"Using OpenAI's GPT3 ({args.model_name}) as generator. The following parameters are ignored: ['decay_constant', 'top_k']")

    model = GPT2Wrapper(model_name=args.model_name, use_cuda=not args.no_cuda) if not args.openai_api_key else args.model_name
    generator = DinoGenerator(
        task_spec=task_specification, model=model, openai_api_key=args.openai_api_key, max_output_length=args.max_output_length,
        decay_constant=args.decay_constant, top_p=args.top_p, top_k=args.top_k, remove_duplicates=args.remove_duplicates,
        remove_identical_pairs=args.remove_identical_pairs, min_num_words=args.min_num_words, min_num_tokens=args.min_num_tokens,
        keep_outputs_without_eos=args.keep_outputs_without_eos, allow_newlines_in_outputs=args.allow_newlines_in_outputs
    )

    print("Starting dataset generation with DINO...")
    outputs = generator.generate_dataset(inputs, num_entries_per_input_and_label=args.num_entries_per_input_and_label,
                                         num_entries_per_label=args.num_entries_per_label, batch_size=args.batch_size)

    print(f"Dataset generation complete, dataset contains {len(outputs)} entries")
    dataset_path = os.path.join(args.output_dir, f'{task_specification["task_name"]}-dataset.jsonl')
    DatasetEntry.save_list(outputs, dataset_path)
    print(f"Done saving dataset to file '{dataset_path}'")
