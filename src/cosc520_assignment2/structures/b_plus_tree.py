"""B+ tree: records live only in leaves; internal keys route searches.

Convention: internal keys[i] equals the minimum key in children[i + 1].
Adjacent leaves are linked through next_leaf in ascending order.
Insertion propagates splits; deletion repairs underflow on the return path.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class BPlusLeaf:
    """Leaf records, with matching sorted keys/values and a forward leaf link."""

    keys: list[int] = field(default_factory=list)
    values: list[int] = field(default_factory=list)
    next_leaf: "BPlusLeaf | None" = None

    def _key_index(self, key: int) -> int:
        """Use the same reverse lower-bound scan as the classic B-tree."""
        i = len(self.keys)
        while i > 0 and key <= self.keys[i - 1]:
            i -= 1
        return i

    def search(self, key: int) -> int | None:
        """Input: key. Output: leaf record's value or None; do not mutate."""
        i = self._key_index(key)
        return self.values[i] if i < len(self.keys) and self.keys[i] == key else None


@dataclass(slots=True)
class BPlusInternal:
    """Routing node: len(children) == len(keys) + 1, with no record values."""

    keys: list[int] = field(default_factory=list)
    children: list["BPlusLeaf | BPlusInternal"] = field(default_factory=list)


class BPlusTree:
    """Integer map; max_keys limits both leaf records and internal separators.

    Non-root leaves have at least ceil(max_keys / 2) records. Non-root internal
    nodes have at least ceil((max_keys + 1) / 2) children. Roots are exceptions.
    """

    def __init__(self, max_keys: int = 63) -> None:
        """Input: odd max_keys >= 3. Output: an empty tree with one leaf root."""
        if max_keys < 3 or max_keys % 2 == 0:
            raise ValueError("max_keys must be odd and at least 3")
        self.max_keys = max_keys
        self.min_leaf_keys = (max_keys + 1) // 2
        self.min_children = (max_keys + 2) // 2
        self.root: BPlusLeaf | BPlusInternal = BPlusLeaf()
        self._size = 0

    def __len__(self) -> int:
        """Input: none. Output: number of leaf records, excluding separators."""
        return self._size

    def _child_index(self, node: BPlusInternal, key: int) -> int:
        """Return the child containing key's range; equality goes right."""
        i = len(node.keys)
        while i > 0 and key < node.keys[i - 1]:
            i -= 1
        return i

    def _find_leaf(self, key: int) -> BPlusLeaf:
        """Input: key. Output: leaf node that may contain the key."""
        curr = self.root
        while isinstance(curr, BPlusInternal):
            curr = curr.children[self._child_index(curr, key)]
        return curr

    def insert(self, key: int, value: int) -> None:
        """Input: key/value. Output: None. Add or replace a leaf record.

        Recursion passes splits up to each parent. Only a root split creates
        a new root and increases the tree height. Duplicate keys update in place.
        """
        split = self._insert_recursive(self.root, key, value)
        if split is not None:
            separator, right = split
            self.root = BPlusInternal(
                keys=[separator], children=[self.root, right],
            )

    def _insert_recursive(
        self, node: BPlusLeaf | BPlusInternal, key: int, value: int,
    ) -> tuple[int, BPlusLeaf | BPlusInternal] | None:
        """Insert into node's subtree and return any split to its caller.

        None means no split needs to be attached to the parent. Otherwise
        return (separator, new_right_node); the original node is kept as the
        left half. Each recursive call remembers its node and child index,
        so it can attach a child's split when that call returns.

        Only the leaf case increments _size, and only for a new key.
        """
        if isinstance(node, BPlusLeaf):
            # Base case: locate the record or its ordered insertion slot.
            i = node._key_index(key)
            if i < len(node.keys) and node.keys[i] == key:
                node.values[i] = value
                return None

            node.keys.insert(i, key)
            node.values.insert(i, value)
            self._size += 1
            if len(node.keys) <= self.max_keys:
                return None

            # Split only after overflow. Copy the right minimum upwards;
            # its actual key/value record must remain in the right leaf.
            middle = len(node.keys) // 2
            right = BPlusLeaf(
                keys=node.keys[middle:], values=node.values[middle:],
                next_leaf=node.next_leaf,
            )
            del node.keys[middle:]
            del node.values[middle:]
            node.next_leaf = right
            return right.keys[0], right

        # Descend once. This frame keeps node and i while the child works.
        i = self._child_index(node, key)
        split = self._insert_recursive(node.children[i], key, value)
        if split is None:
            return None

        # Resume in the parent: the old child remains the left half.
        separator, new_right = split
        node.keys.insert(i, separator)
        node.children.insert(i + 1, new_right)
        if len(node.keys) <= self.max_keys:
            return None

        # Internal split: promote the middle separator, removing it from
        # this level. It equals the minimum of the new right subtree.
        middle = len(node.keys) // 2
        separator = node.keys[middle]
        right_internal = BPlusInternal(
            keys=node.keys[middle + 1:], children=node.children[middle + 1:],
        )
        del node.keys[middle:]
        del node.children[middle + 1:]
        return separator, right_internal

    def search(self, key: int) -> int | None:
        """Input: key. Output: leaf record's value or None; do not mutate.

        Internal separators only route to leaves; they are not stored records.
        """
        return self._find_leaf(key).search(key)

    def delete(self, key: int) -> bool:
        """Remove a leaf record, repair underflows and collapse a one-child root."""
        if not self._delete_recursive(self.root, key):
            return False
        self._size -= 1
        if isinstance(self.root, BPlusInternal) and not self.root.keys:
            self.root = self.root.children[0]
        return True

    @staticmethod
    def _minimum(node: BPlusLeaf | BPlusInternal) -> int:
        """Read a nonempty subtree's minimum for an affected parent boundary."""
        while isinstance(node, BPlusInternal):
            node = node.children[0]
        return node.keys[0]

    def _delete_recursive(self, node: BPlusLeaf | BPlusInternal, key: int) -> bool:
        if isinstance(node, BPlusLeaf):
            i = node._key_index(key)
            if i == len(node.keys) or node.keys[i] != key:
                return False
            node.keys.pop(i)
            node.values.pop(i)
            return True
        i = self._child_index(node, key)
        child = node.children[i]
        if not self._delete_recursive(child, key):
            return False
        # Refresh this boundary BEFORE rotating/merging separators. A non-root
        # leaf has at least two keys before deletion, so its minimum still exists.
        if i > 0:
            node.keys[i - 1] = self._minimum(child)
        minimum = self.min_leaf_keys if isinstance(child, BPlusLeaf) else self.min_children - 1
        if len(child.keys) < minimum:
            self._repair_child(node, i, minimum)
        return True

    def _repair_child(self, parent: BPlusInternal, i: int, minimum: int) -> None:
        """Borrow from a same-parent sibling, otherwise merge adjacent children."""
        child = parent.children[i]
        left = parent.children[i - 1] if i else None
        right = parent.children[i + 1] if i + 1 < len(parent.children) else None
        if left is not None and len(left.keys) > minimum:
            if isinstance(child, BPlusLeaf):
                child.keys.insert(0, left.keys.pop())
                child.values.insert(0, left.values.pop())
                parent.keys[i - 1] = child.keys[0]
            else:
                child.keys.insert(0, parent.keys[i - 1])
                child.children.insert(0, left.children.pop())
                parent.keys[i - 1] = left.keys.pop()
            return
        if right is not None and len(right.keys) > minimum:
            if isinstance(child, BPlusLeaf):
                child.keys.append(right.keys.pop(0))
                child.values.append(right.values.pop(0))
                parent.keys[i] = right.keys[0]
            else:
                child.keys.append(parent.keys[i])
                child.children.append(right.children.pop(0))
                parent.keys[i] = right.keys.pop(0)
            return
        # Prefer merging into the left sibling, if available. Leaf links may
        # cross parents, so next_leaf alone is not sufficient to choose a sibling.
        merge_index = i - 1 if left is not None else i
        left, right = parent.children[merge_index:merge_index + 2]
        separator = parent.keys.pop(merge_index)
        if isinstance(left, BPlusLeaf):
            left.keys.extend(right.keys)
            left.values.extend(right.values)
            left.next_leaf = right.next_leaf
        else:
            left.keys.append(separator)
            left.keys.extend(right.keys)
            left.children.extend(right.children)
        parent.children.pop(merge_index + 1)

    def validate(self) -> None:
        """Check subtree bounds, exact separators, occupancy, depth and leaf chain."""
        seen, depths, leaves = set(), set(), []

        def visit(node, depth):
            if not isinstance(node, (BPlusLeaf, BPlusInternal)) or id(node) in seen:
                raise ValueError("BPlusTree: invalid node, cycle or shared child")
            seen.add(id(node))
            count = len(node.keys)
            minimum = self.min_leaf_keys if isinstance(node, BPlusLeaf) else self.min_children - 1
            if node is self.root:
                minimum = 0 if isinstance(node, BPlusLeaf) else 1
            if not minimum <= count <= self.max_keys:
                raise ValueError("BPlusTree: invalid occupancy")
            if any(a >= b for a, b in zip(node.keys, node.keys[1:])):
                raise ValueError("BPlusTree: keys must be strictly increasing")
            if isinstance(node, BPlusLeaf):
                if len(node.values) != count:
                    raise ValueError("BPlusTree: key/value count mismatch")
                leaves.append(node)
                depths.add(depth)
                return (count, node.keys[0], node.keys[-1]) if count else (0, None, None)
            if len(node.children) != count + 1:
                raise ValueError("BPlusTree: child count mismatch")
            bounds = [visit(child, depth + 1) for child in node.children]
            if node.keys != [low for _, low, _ in bounds[1:]]:
                raise ValueError("BPlusTree: separator differs from right subtree minimum")
            if any(a[2] >= b[1] for a, b in zip(bounds, bounds[1:])):
                raise ValueError("BPlusTree: overlapping child ranges")
            return sum(b[0] for b in bounds), bounds[0][1], bounds[-1][2]

        total, _, _ = visit(self.root, 0)
        if total != self._size or len(depths) != 1:
            raise ValueError("BPlusTree: size mismatch or unequal leaf depths")
        for i, leaf in enumerate(leaves):
            expected = leaves[i + 1] if i + 1 < len(leaves) else None
            if leaf.next_leaf is not expected:
                raise ValueError("BPlusTree: leaf chain differs from tree order")
