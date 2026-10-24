"""Shared semantics for all three integer-to-integer ordered maps."""

from typing import Protocol


class OrderedMap(Protocol):
    """Use unique integer keys; benchmark values are key // 2.

    Duplicate insertions replace a value without changing the size. The main
    benchmark inserts only new keys. None means absence; zero is a valid value.
    Implementations own their nodes and must not delegate storage/search to dict,
    set, or a library that implements an ordered map. Lists for node arrays are OK.
    """

    def insert(self, key: int, value: int) -> None:
        """Input: integer key/value. Output: None. Insert or replace one mapping."""
        ...

    def search(self, key: int) -> int | None:
        """Input: integer key. Output: its value, or None. Do not mutate the map."""
        ...

    def delete(self, key: int) -> bool:
        """Input: integer key. Output: True if removed, False if already absent."""
        ...

    def __len__(self) -> int:
        """Input: none. Output: number of distinct stored keys, in constant time."""
        ...

    def validate(self) -> None:
        """Input: none. Output: None; raise ValueError for a broken invariant.

        This diagnostic must not change the map. It is never a timed operation.
        Benchmark validation is optional because a full traversal can be costly.
        """
        ...
