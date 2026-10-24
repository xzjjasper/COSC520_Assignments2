"""Only the three student structures are available to the benchmark CLI."""

from .interfaces import OrderedMap
from .structures import BTree, BPlusTree, SkipList

STRUCTURES = ("btree", "bplus", "skiplist")


def create_structure(name: str, *, max_keys: int = 63, probability: float = 0.5,
                     max_level: int = 32, seed: int = 42) -> OrderedMap:
    """Input: name and parameters. Output: a new empty student structure."""
    if name == "btree":
        return BTree(max_keys=max_keys)
    if name == "bplus":
        return BPlusTree(max_keys=max_keys)
    if name == "skiplist":
        return SkipList(probability=probability, max_level=max_level, seed=seed)
    raise ValueError(f"unknown structure: {name}")
