import json
from typing import *
import logging
import random
import re


def parse_data(path: str) -> List[Dict[str, Any]]:
    """
    Read the JSON file at <path> and return the relevant information.
    """
    with open(path) as f:
        data = json.load(f)
        documents = data["documents"]
        logging.info(f"Loaded JSON file containing {len(documents)} documents.")

        return documents


def display_data(data: Any, depth: int = 0) -> str:
    """
    Convert the given data into a human readable string.
    """
    lines = []

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{' ' * depth}{key}:")
                lines.append(display_data(value, depth + 1))
            elif isinstance(value, list):
                if value:
                    lines.append(f"{' ' * depth}{key}:")

                for i, item in enumerate(value):
                    lines.append(f"{' ' * (depth + 1)}#{i + 1}")
                    lines.append(display_data(item, depth + 1))
            else:
                lines.append(f"{' ' * depth}{key}: {value}")

    else:
        return f"{' ' * depth}{data}"

    if lines:
        final_lines = []
        for line in lines:
            if line:
                final_lines.append(line)

        return "\n".join(final_lines)
    else:
        return ""


def sample_and_view(documents: List[Dict[str, Any]]) -> None:
    """
    Randomly sample a document from <documents> and print the data to human-readable format.
    """
    document = random.choice(documents)
    class_doc = PaperDocument(document)

    print(class_doc)


class Field:
    """
    Represents a field from the NLM Chem dataset. This class extracts the information from the JSON data, and needs
    the given input to follow a specific format.
    """
    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Create an field, recursively if necessary.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                value_infon = Field(value)
                setattr(self, key, value_infon)

            elif isinstance(value, list):
                self.process_list_value(key, value)

            else:
                setattr(self, key, value)

    def process_list_value(self, key: str, value: List[Any]) -> None:
        """
        Handle the case where the value of a key in the given field is a list.
        """
        # list should have the same data type throughout
        if value and any(type(item) != type(value[0]) for item in value[1:]):
            raise TypeError("Expected uniform typing throughout the list, got mismatches.")

        if value and isinstance(value[0], dict):
            setattr(self, key, [Field(item) for item in value])

        else:
            setattr(self, key, value)

    def _str(self, depth: int = 0, tab_size: int = 2) -> str:
        """
        Return a string representation of the current field indented based on <depth>.
        """
        num_space = depth * tab_size
        indent = " " * num_space

        lines = []

        for key, value in vars(self).items():
            if isinstance(value, Field):
                if value.is_empty():
                    empty_brackets = "{}"
                    lines.append(f"{indent}{key}: {empty_brackets}")
                else:
                    lines.append(f"{indent}{key}:")
                    lines.append(value._str(depth=depth + 1))

            elif isinstance(value, list):
                if not value:
                    lines.append(f"{indent}{key}: []")

                else:
                    lines.append(f"{indent}{key}:")

                    for i, item in enumerate(value):
                        if len(value) > 1:
                            lines.append(f"{indent}{' ' * tab_size}#{key}-{i + 1}")
                        lines.append(str(item) if not isinstance(item, Field) else item._str(depth=depth + 1))

                        if i < len(value) - 1:
                            lines.append("\n")

            else:
                lines.append(f"{indent}{key}: {value}")

        text = "\n".join(lines)

        return re.sub("\n{3,}", "\n\n", text)

    def is_empty(self) -> bool:
        """
        Return True if this field contains no data, False otherwise.
        """
        return vars(self) == {}

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the current field.
        """
        return self._str()

    def __repr__(self) -> str:
        return str(self)


class PaperDocument:
    """
    Wraps the Field class to represent an entire document with multiple fields.
    """
    def __init__(self, document: Dict[str, Any]) -> None:
        """
        Initializes a document.
        """
        self.data = Field(document)

    def __str__(self) -> str:
        return str(self.data)


if __name__ == "__main__":
    path = "/home/haonguyen/nlp_project/Track-2_NLM-Chem/BC7T2-CDR-corpus-train.BioC.json"
    docs = parse_data(path)
    sample_and_view(docs)
