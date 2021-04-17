from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, PreTrainedTokenizer, PreTrainedModel

from generation import SelfDebiasingGPT2LMHeadModel
from utils import DatasetEntry

PLACEHOLDER_STR = "<X1>"


class DinoGenerator:
    """
    This class represents a generative language model which can be used to generate datasets from instructions.
    """

    def __init__(self, model: 'ModelWrapper', task_spec: Dict[str, Any], max_output_length: int = 40, decay_constant: float = 100,
                 top_p: float = 0.9, top_k: int = 5, remove_duplicates: bool = True, remove_identical_pairs: bool = False,
                 min_num_words: int = -1, keep_outputs_without_eos: bool = False, allow_newlines_in_outputs: bool = False):
        """
        :param model: a wrapper around the underlying language model
        :param task_spec: the task specification
        :param max_output_length: the maximum output length for each generated text
        :param decay_constant: the decay constant for self-debiasing
        :param top_p: p value for top-p sampling (set to 0 to perform no top-p sampling)
        :param top_k: k value for top-k sampling (set to 0 to perform no top-k sampling)
        :param remove_duplicates: whether duplicates should be removed from the generated dataset
        :param remove_identical_pairs: whether text pairs with identical texts should be removed (only for text pair datasets)
        :param min_num_words: the minimum number of (whitespace-separated) words for each dataset entry
        :param keep_outputs_without_eos: if set to true, examples where the language model does not output a quotation mark (which is
               interpreted as a signal that it has completed its output) are not removed from the dataset.
        :param allow_newlines_in_outputs: if set to true, model outputs that contain a newline character before the end-of-sequence token
               (a quotation mark) are not removed from the dataset
        """
        self.model = model
        self.max_output_length = max_output_length
        self.decay_constant = decay_constant
        self.top_p = top_p
        self.top_k = top_k
        self.remove_duplicates = remove_duplicates
        self.remove_identical_pairs = remove_identical_pairs
        self.min_num_words = min_num_words
        self.keep_outputs_without_eos = keep_outputs_without_eos
        self.allow_newlines_in_outputs = allow_newlines_in_outputs

        self.labels = list(task_spec['labels'].keys())
        self.instructions = {label: task_spec['labels'][label]['instruction'] for label in self.labels}
        self.counter_labels = {label: task_spec['labels'][label]['counter_labels'] for label in self.labels}

    def generate_dataset(self, input_texts: Optional[List[str]], num_entries_per_input_and_label: Optional[int] = None,
                         num_entries_per_label: Optional[int] = None) -> List[DatasetEntry]:
        """
        Generate a new dataset.
        :param input_texts: an optional list of raw texts; this is required for generating text pair datasets
        :param num_entries_per_input_and_label: the number of entries to generate for each pair of input text and label
        :param num_entries_per_label: the number of entries to generate for each label
        :return: the generated dataset
        """

        generate_with_inputs = input_texts is not None

        if not generate_with_inputs:
            input_texts = list(range(num_entries_per_label))
            num_entries_per_input_and_label = 1

        input_iterator = tqdm(input_texts, desc="Dataset Entries")
        dataset = []

        for input_text_or_id in input_iterator:
            for label in self.labels:
                dataset += self._generate_dataset_entries(input_text_or_id, label=label, num_entries=num_entries_per_input_and_label,
                                                          generate_with_inputs=generate_with_inputs)

        dataset = self._postprocess_dataset(dataset, generate_with_inputs)
        return dataset

    def _generate_dataset_entries(self, input_text_or_id: Union[str, int], label: str, num_entries: int,
                                  generate_with_inputs: bool) -> List[DatasetEntry]:

        instruction = self._build_instruction(label, input_text_or_id, generate_with_inputs)
        counter_instructions = [
            self._build_instruction(other_label, input_text_or_id, generate_with_inputs) for other_label in self.counter_labels[label]
        ]

        model_outputs = self.model.generate_self_debiasing(
            input_text=instruction, debiasing_texts=counter_instructions, num_samples=num_entries, decay_constant=self.decay_constant,
            do_sample=True, min_length=self.max_output_length, max_length=self.max_output_length, top_k=self.top_k, top_p=self.top_p
        )

        model_outputs = [
            self._process_output(input_text=input_text_or_id, output_text=output, label=label, generate_with_inputs=generate_with_inputs)
            for output in model_outputs
        ]

        model_outputs = [output for output in model_outputs if output is not None]
        return model_outputs

    def _build_instruction(self, label: str, text: str, generate_with_inputs: bool) -> str:
        instruction_template = self.instructions[label]

        if generate_with_inputs:
            assert instruction_template.count(PLACEHOLDER_STR) == 1, \
                f"An input text was provided, but the instruction for label '{label}' does not contain exactly one placeholder"
            return instruction_template.replace(PLACEHOLDER_STR, text)
        else:
            assert instruction_template.count(PLACEHOLDER_STR) == 0, \
                f"No input text was provided, but the instruction for label '{label}' contains a placeholder"
            return instruction_template

    def _process_output(self, input_text: Union[str, int], output_text: str, label: str, generate_with_inputs: bool) \
            -> Optional[DatasetEntry]:
        output_text = output_text.split('"')[0] if '"' in output_text else (output_text if self.keep_outputs_without_eos else None)
        if output_text and ('\n' not in output_text or self.allow_newlines_in_outputs):
            text_a = input_text if generate_with_inputs else output_text
            text_b = output_text if generate_with_inputs else None
            return DatasetEntry(text_a=text_a, text_b=text_b, label=label)
        return None

    def _postprocess_dataset(self, dataset: List[DatasetEntry], generate_with_inputs: bool) -> List[DatasetEntry]:

        if self.remove_duplicates:
            dataset = list(set(dataset))

        if self.min_num_words > 0:
            if generate_with_inputs:
                dataset = [entry for entry in dataset if len(entry.text_b.split()) >= self.min_num_words]
            else:
                dataset = [entry for entry in dataset if len(entry.text_a.split()) >= self.min_num_words]

        if generate_with_inputs and self.remove_identical_pairs:
            dataset = [entry for entry in dataset if entry.text_a != entry.text_b]

        return dataset


class ModelWrapper(ABC):
    """
    This class represents a wrapper for a pretrained language model that provides high-level functions for the generation of texts with
    the self-debiasing method described in https://arxiv.org/abs/2103.00453.
    """

    def __init__(self, use_cuda: bool = True):
        """
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = None  # type: Optional[PreTrainedTokenizer]
        self._model = None  # type: Optional[PreTrainedModel]

    def query_model(self, input_text: str) -> torch.FloatTensor:
        """For a given input text, returns the probability distribution over possible next tokens."""
        return self.query_model_batch([input_text])[0]

    @abstractmethod
    def query_model_batch(self, input_texts: List[str]) -> torch.FloatTensor:
        """For a batch of input texts, returns the probability distribution over possible next tokens."""
        pass

    @abstractmethod
    def generate(self, input_text: str, **kwargs) -> str:
        """Generates a continuation for a given input text."""
        pass

    @abstractmethod
    def generate_self_debiasing(self, input_text: str, debiasing_texts: List[str], num_samples: int = 1, decay_constant: float = 100,
                                epsilon: float = 0.01, debug: bool = False, **kwargs) -> List[str]:
        """
        Generates continuations for the given input texts with self-debiasing.
        :param input_texts: the input texts to generate continuations for
        :param debiasing_prefixes: the debiasing prefixes to be used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        :param debug: whether to print additional debugging output
        :param kwargs: further arguments are passed on to the original generate function
        :return: the list of generated continuations
        """
        pass


class GPT2Wrapper(ModelWrapper):

    def __init__(self, model_name: str = "gpt2-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__(use_cuda=use_cuda)
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(model_name)  # type: SelfDebiasingGPT2LMHeadModel
        if use_cuda:
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

    def query_model_batch(self, input_texts: List[str]):
        inputs = self._tokenizer.batch_encode_plus(input_texts, padding=True, max_length=512, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1
        output = self._model(**inputs)['logits']
        return torch.stack([output[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])

    def generate(self, input_text: str, **kwargs):
        input_ids = self._tokenizer.encode(input_text, return_tensors='pt').to(self._device)
        output_ids = self._model.generate(input_ids, **kwargs)[0]
        return self._tokenizer.decode(output_ids)

    def generate_self_debiasing(self, input_text: str, debiasing_texts: List[str], num_samples: int = 1, decay_constant: float = 100,
                                epsilon: float = 0.01, debug: bool = False, min_length: int = None, max_length: int = None,
                                **kwargs) -> List[str]:

        self._model.init_logits_processor(num_debiasing_prefixes=len(debiasing_texts), decay_constant=decay_constant, epsilon=epsilon,
                                          debug=debug, tokenizer=self._tokenizer)

        inputs = [input_text] * num_samples
        for debiasing_text in debiasing_texts:
            inputs += [debiasing_text] * num_samples

        inputs = self._tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(**inputs, min_length=min_length, max_length=max_length, **kwargs)

        batch_size = output_ids.shape[0] // (1 + len(debiasing_texts))
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]
        return self._tokenizer.batch_decode(output_ids)
