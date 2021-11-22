from typing import Set, List, Any, Tuple
from utils import parse_data, PaperDocument
from typing import *
from collections import defaultdict
from tqdm.auto import tqdm
import re
from string import punctuation
import os
import json
from argparse import ArgumentParser


class Entity:
    def __init__(self, text: str) -> None:
        self.full_text = text

        self.num_words = len(words)
        self.first_word = words[0]
        self.remaining = words[1:]
        self.len_remaining = len(self.remaining)

    def matches_entity(self, beginning: str, remaining: List[str]) -> bool:
        """
        Using the given beginning word and the remaining words to return True if the full string matches with this
        current entity, and False otherwise.
        """
        beginning = beginning.strip(punctuation)
        remaining = [word.strip(punctuation) for word in remaining]
        return beginning == self.first_word and remaining == self.remaining


def get_targets(documents: List[Dict[str, Any]], topics: bool = False) -> List[Tuple[Any, List[str]]]:
    """
    Extract the named entities that should be recognized from each document's text, and create the labels for it.

    If topics is False, the labels are simply all the named entities (all chemicals). Otherwise, only the topic
    chemicals will be given.
    """
    data = []

    for document in tqdm(documents, desc="Processing NER targets..."):
        doc_class = PaperDocument(document)

        # get the abstract document -- should be the second passage
        abstract_text = doc_class.data.passages[1].text
        annotations = doc_class.data.passages[1].annotations

        labels = set()

        for annotation in annotations:
            if annotation.text == "":
                continue

            if not topics:
                labels.add(annotation.text.lower())

            elif topics and annotation.infons.type == "MeSH_Indexing_Chemical":
                labels.add(annotation.text.lower())

        data.append((abstract_text, list(labels)))

    return data


def load_and_write_data(input_path: str, output_dir: str, base_output_name: str, prompt_start: str, prompt_end: str) \
        -> None:
    """
    Load the data from the path and write the output to the desired output directory.
    """
    data_corpus = parse_data(input_path)
    data_with_targets = get_targets(data_corpus)

    output_data_file = open(os.path.join(output_dir, f"{base_output_name}_full_text.json"), "w+")

    for entry in data_with_targets:
        inputs, labels = entry

        labels_text = ", ".join(f"\"{item}\"" for item in labels if item != "")

        data_entry = {"text": f"<|startoftext|> {prompt_start} {inputs} \n===\n{prompt_end} {labels_text} <|endoftext|>"}
        json.dump(data_entry, output_data_file)
        output_data_file.write("\n")

    output_data_file.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_file", help="Path(s) to JSON files holding the raw data.", type=str, nargs="+",
                        required=True)
    parser.add_argument("--output_dir", help="Path to write the processed dataset(s) to.", type=str, required=True)
    parser.add_argument("--prompt_start", help="String to add to the beginning of the text for prompting purposes.",
                        type=str, required=True)
    parser.add_argument("--prompt_end", help="String to add to the end of the text for prompting purposes", type=str,
                        default="")

    args = parser.parse_args()

    for input_file in args.source_file:
        last_slash = input_file.rfind("/")

        if last_slash == -1:
            filename_leaf = input_file

        else:
            filename_leaf = input_file[last_slash + 1:]

        base_name = filename_leaf[:filename_leaf.find(".")]

        load_and_write_data(input_file, args.output_dir, base_name, args.prompt_start, args.prompt_end)
