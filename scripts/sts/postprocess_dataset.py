import argparse
import random
from collections import defaultdict
from typing import List

from utils import DatasetEntry


def postprocess_dataset(
        dataset: List[DatasetEntry],
        remove_identical_pairs: bool = True,
        remove_duplicates: bool = True,
        add_sampled_pairs: bool = True,
        max_num_text_b_for_text_a_and_label: int = 2,
        label_smoothing: float = 0.2,
        seed: int = 42
) -> List[DatasetEntry]:
    """
    Apply various postprocessing steps to a STS dataset.
    :param dataset: The dataset to postprocess.
    :param remove_identical_pairs: If set to true, we remove all pairs (x1, x2, y) where x1 == x2 as a bi-encoder cannot learn from them.
    :param remove_duplicates:  If set to true, if there are pairs (x1, x2, y) and (x1', x2', y') with x1 == x1', x2 == x2', y == y', we
           only keep one of them.
    :param add_sampled_pairs: If set to true, we add pairs of randomly sampled x1's and x2's and similarity 0 to the dataset.
    :param max_num_text_b_for_text_a_and_label: We keep at most this many examples for each pair of text_a and similarity label.
    :param label_smoothing: The amount of label smoothing to apply.
    :param seed: The seed for the random number generator used to shuffle the dataset and for adding sampled pairs.
    :return: The postprocessed dataset.
    """
    postprocessed_dataset = []
    num_text_b_for_text_a_and_label = defaultdict(int)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    if remove_duplicates:
        dataset = list(set(dataset))

    for example in dataset:
        if remove_identical_pairs and example.text_a == example.text_b:
            continue

        example.label = example.label * (1 - label_smoothing) + (label_smoothing / 3 * 1.5)

        if max_num_text_b_for_text_a_and_label > 0:
            if num_text_b_for_text_a_and_label[(example.text_a, example.label)] >= max_num_text_b_for_text_a_and_label:
                continue
        postprocessed_dataset.append(example)
        num_text_b_for_text_a_and_label[(example.text_a, example.label)] += 1

    if add_sampled_pairs:
        sampled_dataset = []

        for text_a in set(x.text_a for x in postprocessed_dataset):
            for _ in range(max_num_text_b_for_text_a_and_label):
                text_b = rng.choice(postprocessed_dataset).text_b
                sampled_dataset.append(DatasetEntry(text_a=text_a, text_b=text_b, label=0))

        postprocessed_dataset += sampled_dataset
    return postprocessed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", type=str, required=True,
                        help="The input file which contains the STS dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The output file to which the postprocessed STS dataset is saved")

    args = parser.parse_args()

    ds = DatasetEntry.read_list(args.input_file)
    ds_pp = postprocess_dataset(ds)
    DatasetEntry.save_list(ds_pp, args.output_file)
