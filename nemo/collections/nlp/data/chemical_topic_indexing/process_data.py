from utils import parse_data, PaperDocument
from typing import *
from collections import defaultdict
from tqdm.auto import tqdm
import re
from string import punctuation


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


def get_ner_targets(documents: List[Dict[str, Any]]) -> List[Tuple[str, List[str], List[str]]]:
    """
    Extract the named entities that should be recognized from each document's text, and create the labels for it.

    Each token will be classified as either "O", "I" or "B":
        - "O" is assigned to a token if it is not part of an entity
        - "I" is assigned to a token if it is part of an entity and the previous token is not part of one or part of
          the same entity
        - "B" is assigned to a token if it is part of an entity and the previous token is part of a different entity
    """
    data = []

    for document in tqdm(documents, desc="Processing NER targets..."):
        doc_class = PaperDocument(document)
        # print(doc_class)

        # get the abstract document -- should be the second passage
        abstract_text = doc_class.data.passages[1].text
        annotations = doc_class.data.passages[1].annotations

        entities_start = defaultdict(set)
        entities = set()

        for annotation in annotations:
            entity_text = annotation.text
            if entity_text:
                entities.add(entity_text)
                first_word = entity_text.split()[0]

                entity = Entity(entity_text)
                entities_start[first_word].add(entity)

        tokens = abstract_text.split()

        labels = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # the token matches with the start of an entity
            if token in entities_start:
                # we need to check if this token actually belongs to entity
                # we stored a map of first entity's words -> rest of entity -- so iterate through the entities that
                # start with this token
                entity_matched = False
                for entity in entities_start[token]:
                    # number of words after this token for comparison
                    len_remaining = entity.len_remaining
                    remaining = tokens[i + 1: i + len_remaining + 1]     # retrieve those words

                    text_matches_entity = entity.matches_entity(token, remaining)     # perform the matching

                    # if this phrase matches
                    if text_matches_entity:
                        # if the previous token is part of an entity then we need to signal that this token is the start
                        # of a new entity
                        if labels and labels[-1] in ["B", "I"]:
                            labels.append("B")
                            labels.extend(["I" for _ in range(len_remaining)])

                        # if there are no entities yet, or the previous token is not an entity
                        else:
                            labels.extend(["I" for _ in range(entity.num_words)])

                        entity_matched = True

                        # move the index to after the end of the current entity to continue looking for entities
                        i += entity.num_words

                        # if we have found a matching entity starting with the current token, we know that it will
                        # not match with any other entity, so we do not need to keep searching
                        break

                # does not belong to an entity
                if not entity_matched:
                    labels.append("O")
                    i += 1

            # this token is not the first word of any entity -- no need for searching, we know that this isn't part of
            # any entity since if it's a word in the middle of an entity, we would've covered it in the code above and
            # moved the index passed it
            else:
                labels.append("O")
                i += 1

        data.append((abstract_text, entities, labels))

    for entry in data:
        print("Entities:", entry[1])
        for word, label in zip(entry[0].split(), entry[2]):
            print(f"{word} | label: {label}")
        print("\n**************\n")

    return data


if __name__ == "__main__":
    data = parse_data("/home/haonguyen/nlp_project/Track-2_NLM-Chem/BC7T2-CDR-corpus-train.BioC.json")
    get_ner_targets(data)
