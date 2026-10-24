"""Randomized ordered map with independent geometric node heights."""

from dataclasses import dataclass
import random


@dataclass(slots=True)
class SkipListNode:
    """One mapping plus its forward links; len(forward) is the node's height.

    Level 0 contains every record. The sentinel alone uses key/value=None.
    Links at every level must point to strictly larger keys.
    """

    key: int | None
    value: int | None
    forward: list["SkipListNode | None"]


class SkipList:
    """Integer map with private RNG, independent of workload generation."""

    def __init__(self, probability: float = 0.5, max_level: int = 32,
                 seed: int = 42) -> None:
        """Input: 0 < probability < 1, positive max_level, RNG seed.

        Output: empty map with a sentinel; level is the active level count.
        """
        if not 0 < probability < 1 or max_level < 1:
            raise ValueError("require 0 < probability < 1 and max_level >= 1")
        self.probability = probability
        self.max_level = max_level
        self.rng = random.Random(seed)
        self.head = SkipListNode(None, None, [None] * max_level)
        self.level = 1
        self._size = 0

    def __len__(self) -> int:
        """Input: none. Output: number of records, excluding the sentinel."""
        return self._size

    def _random_level(self) -> int:
        """Draw a capped geometric height; called inside timed insertion."""
        height = 1
        while height < self.max_level and self.rng.random() < self.probability:
            height += 1
        return height

    def _predecessors(self, key: int) -> list[SkipListNode]:
        """Remember the last node strictly below key on each level."""
        update = [self.head] * self.max_level
        curr = self.head
        for level in range(self.level - 1, -1, -1):
            while curr.forward[level] is not None and curr.forward[level].key < key:
                curr = curr.forward[level]
            update[level] = curr
        return update

    def insert(self, key: int, value: int) -> None:
        """Add or replace; duplicate updates do not consume randomness."""
        update = self._predecessors(key)
        existing = update[0].forward[0]
        if existing is not None and existing.key == key:
            existing.value = value
            return
        height = self._random_level()
        node = SkipListNode(key, value, [None] * height)
        for level in range(height):
            node.forward[level] = update[level].forward[level]
            update[level].forward[level] = node
        self.level = max(self.level, height)
        self._size += 1

    def search(self, key: int) -> int | None:
        """Descend from the highest active level without allocating a path."""
        curr = self.head
        for level in range(self.level - 1, -1, -1):
            while curr.forward[level] is not None and curr.forward[level].key < key:
                curr = curr.forward[level]
        curr = curr.forward[0]
        return curr.value if curr is not None and curr.key == key else None

    def delete(self, key: int) -> bool:
        """Unlink the record at every height and drop empty upper levels."""
        update = self._predecessors(key)
        node = update[0].forward[0]
        if node is None or node.key != key:
            return False
        for level in range(len(node.forward)):
            update[level].forward[level] = node.forward[level]
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        self._size -= 1
        return True

    def validate(self) -> None:
        """Check order, heights and exact membership at every level; preserve RNG."""
        if (self.head.key is not None or self.head.value is not None
                or len(self.head.forward) != self.max_level
                or not 1 <= self.level <= self.max_level):
            raise ValueError("SkipList: invalid sentinel or active level count")
        expected = [[] for _ in range(self.max_level)]
        seen, previous, height = set(), None, 1
        curr = self.head.forward[0]
        while curr is not None:
            if not isinstance(curr, SkipListNode) or id(curr) in seen:
                raise ValueError("SkipList: invalid node or cycle")
            seen.add(id(curr))
            if (curr.key is None or curr.value is None
                    or not 1 <= len(curr.forward) <= self.max_level
                    or (previous is not None and curr.key <= previous)):
                raise ValueError("SkipList: invalid record, ordering or height")
            height = max(height, len(curr.forward))
            for level in range(len(curr.forward)):
                expected[level].append(id(curr))
            previous, curr = curr.key, curr.forward[0]
        if len(seen) != self._size or height != self.level:
            raise ValueError("SkipList: size or active level mismatch")
        for level in range(self.max_level):
            curr = self.head.forward[level]
            for identity in expected[level]:
                if id(curr) != identity:
                    raise ValueError("SkipList: incorrect level membership")
                curr = curr.forward[level]
            if curr is not None:
                raise ValueError("SkipList: extra link or cycle")
