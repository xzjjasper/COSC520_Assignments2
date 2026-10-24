"""Classic B-tree: records may live in internal nodes or leaves.

Insertion and deletion prepare children before descending.
For minimum degree t, non-root nodes contain t-1 through 2t-1 keys.
All leaves have the same depth; an internal node has len(keys) + 1 children.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class BTreeNode:
    """One node; keys[i] and values[i] are a complete mapping.

    keys are strictly increasing. Leaf children are empty. Internal children
    partition the key space around the records stored in this node.
    """

    leaf: bool = True
    keys: list[int] = field(default_factory=list)
    values: list[int] = field(default_factory=list)
    children: list["BTreeNode"] = field(default_factory=list)

    def _key_index(self, key: int) -> int:
        """Return the first index whose key is >= key, or len(self.keys).

        Use the same reverse lower-bound scan as BPlusLeaf. Search, location,
        insertion and deletion share this node-local lookup.
        """
        i = len(self.keys)
        while i > 0 and key <= self.keys[i - 1]:
            i -= 1
        return i

    def search(self, key: int) -> int | None:
        """Return a record's value from this subtree, or None; do not mutate.

        Unlike a B+ internal separator, an internal B-tree key has a value.
        Check this node first; descend only when the record is absent here.
        """
        i = self._key_index(key)
        if i < len(self.keys) and self.keys[i] == key:
            return self.values[i]
        if self.leaf:
            return None
        return self.children[i].search(key)


class BTree:
    """Ordered integer map with a configurable odd maximum key capacity."""

    def __init__(self, max_keys: int = 63) -> None:
        """Input: odd max_keys >= 3. Output: an empty tree with one leaf root.

        This variant uses minimum degree t >= 2: max_keys = 2*t - 1.
        Non-root nodes hold at least t-1 keys; non-root internal nodes
        therefore have at least t children. A full node can be split before
        insertion: promote its middle key and leave t-1 keys on each side.
        Odd capacity is a choice of this variant, not a rule for all B-trees.
        """
        if max_keys < 3 or max_keys % 2 == 0:
            raise ValueError("max_keys must be odd and at least 3")
        self.max_keys = max_keys
        self.min_degree = (max_keys + 1) // 2
        self.root = BTreeNode()
        self._size = 0

    def __len__(self) -> int:
        """Input: none. Output: distinct key count; maintained by insert/delete."""
        return self._size

    def _find_key_position(self, key: int) -> tuple[BTreeNode, int]:
        """Return (node, index) for an existing key or a missing key's leaf slot.

        If found, node.keys[index] == key; node may be internal or a leaf.
        Otherwise node is a leaf and index is in 0..len(node.keys): insert
        before node.keys[index], or append when index == len(node.keys).
        An empty tree returns (self.root, 0).

        Test for a match with:
            index < len(node.keys) and node.keys[index] == key

        The tree is not changed. The leaf may already be full, so the caller
        must handle splitting; a split may invalidate the returned position.
        This returns a local slot, not the global predecessor and successor
        (either of those keys can be stored in an ancestor).
        """
        curr = self.root
        while True:
            i = curr._key_index(key)

            # Classic B-trees store actual records in internal nodes too.
            if i < len(curr.keys) and curr.keys[i] == key:
                return curr, i
            if curr.leaf:
                return curr, i

            # Missing here: children[i] covers the interval containing key,
            # including the intervals before the first and after the last key.
            curr = curr.children[i]

    def insert(self, key: int, value: int) -> None:
        """Input: key/value. Output: None. Add or replace a mapping.

        Split a full root first, then split full children before descending.
        Every node entered has room for a record promoted by its child.
        Only inserting a new leaf record increases _size; moving records or
        updating duplicates does not. Preemptive splits can change the tree's
        shape even when the operation ultimately updates an existing key.
        """
        if len(self.root.keys) == self.max_keys:
            self.root = BTreeNode(leaf=False, children=[self.root])
            self._split_child(self.root, 0)

        curr = self.root
        while True:
            i = curr._key_index(key)
            if i < len(curr.keys) and curr.keys[i] == key:
                # A classic B-tree may hold the existing record internally.
                curr.values[i] = value
                return

            if curr.leaf:
                curr.keys.insert(i, key)
                curr.values.insert(i, value)
                self._size += 1
                return

            if len(curr.children[i].keys) == self.max_keys:
                self._split_child(curr, i)
                # The promoted record now separates the two child halves.
                if key == curr.keys[i]:
                    curr.values[i] = value
                    return
                if key > curr.keys[i]:
                    i += 1
            curr = curr.children[i]

    def _split_child(self, parent: BTreeNode, index: int) -> None:
        """Split a full child of a non-full internal parent, without adding data.

        Preconditions: parent has room for one key; children[index] has 2*t-1
        keys. Move its middle key AND value into parent. Keep the old child
        as the left half; insert the new right half immediately after it.
        Both halves retain t-1 keys and, for internal nodes, t children.
        """
        child = parent.children[index]
        t = self.min_degree
        right = BTreeNode(
            leaf=child.leaf,
            keys=child.keys[t:],
            values=child.values[t:],
            children=child.children[t:] if not child.leaf else [],
        )

        # Capture the median record before trimming the original child.
        parent.keys.insert(index, child.keys[t - 1])
        parent.values.insert(index, child.values[t - 1])
        parent.children.insert(index + 1, right)
        del child.keys[t - 1:]
        del child.values[t - 1:]
        if not child.leaf:
            del child.children[t:]

    def search(self, key: int) -> int | None:
        """Delegate to the root: records may be in internal nodes or leaves."""
        return self.root.search(key)

    def delete(self, key: int) -> bool:
        """Remove a record, restoring occupancy and shrinking an empty root.

        Preparing a child can change tree shape even for a missing key.
        Only a successful removal decreases the distinct record count.
        """
        removed = self._delete_recursive(self.root, key)
        if not self.root.leaf and not self.root.keys:
            self.root = self.root.children[0]
        if removed:
            self._size -= 1
        return removed

    def _delete_recursive(self, node: BTreeNode, key: int) -> bool:
        """Descend into children with at least t keys, so one can be removed."""
        i = node._key_index(key)
        if i < len(node.keys) and node.keys[i] == key:
            if node.leaf:
                node.keys.pop(i)
                node.values.pop(i)
                return True
            left, right = node.children[i:i + 2]
            if len(left.keys) >= self.min_degree:
                predecessor = left
                while not predecessor.leaf:
                    predecessor = predecessor.children[-1]
                node.keys[i], node.values[i] = predecessor.keys[-1], predecessor.values[-1]
                return self._delete_recursive(left, node.keys[i])
            if len(right.keys) >= self.min_degree:
                successor = right
                while not successor.leaf:
                    successor = successor.children[0]
                node.keys[i], node.values[i] = successor.keys[0], successor.values[0]
                return self._delete_recursive(right, node.keys[i])
            merged = self._merge_children(node, i)
            return self._delete_recursive(merged, key)

        if node.leaf:
            return False
        if len(node.children[i].keys) == self.min_degree - 1:
            if i > 0 and len(node.children[i - 1].keys) >= self.min_degree:
                self._borrow_left(node, i)
            elif i < len(node.keys) and len(node.children[i + 1].keys) >= self.min_degree:
                self._borrow_right(node, i)
            elif i < len(node.keys):
                self._merge_children(node, i)
            else:
                self._merge_children(node, i - 1)
                i -= 1
        return self._delete_recursive(node.children[i], key)

    def _borrow_left(self, parent: BTreeNode, i: int) -> None:
        """Rotate the parent record down and the left sibling's maximum up."""
        child, left = parent.children[i], parent.children[i - 1]
        child.keys.insert(0, parent.keys[i - 1])
        child.values.insert(0, parent.values[i - 1])
        parent.keys[i - 1], parent.values[i - 1] = left.keys.pop(), left.values.pop()
        if not child.leaf:
            child.children.insert(0, left.children.pop())

    def _borrow_right(self, parent: BTreeNode, i: int) -> None:
        """Rotate the parent record down and the right sibling's minimum up."""
        child, right = parent.children[i], parent.children[i + 1]
        child.keys.append(parent.keys[i])
        child.values.append(parent.values[i])
        parent.keys[i], parent.values[i] = right.keys.pop(0), right.values.pop(0)
        if not child.leaf:
            child.children.append(right.children.pop(0))

    def _merge_children(self, parent: BTreeNode, i: int) -> BTreeNode:
        """Join two minimum-size children with the parent's intervening record."""
        left, right = parent.children[i:i + 2]
        left.keys.append(parent.keys.pop(i))
        left.values.append(parent.values.pop(i))
        left.keys.extend(right.keys)
        left.values.extend(right.values)
        left.children.extend(right.children)
        parent.children.pop(i + 1)
        return left

    def validate(self) -> None:
        """Check order, occupancy, ranges, equal depth and size without mutation."""
        seen, depths = set(), set()

        def visit(node, depth, lower=None, upper=None):
            if not isinstance(node, BTreeNode) or id(node) in seen:
                raise ValueError("BTree: invalid node, cycle or shared child")
            seen.add(id(node))
            count = len(node.keys)
            minimum = (0 if node.leaf else 1) if node is self.root else self.min_degree - 1
            if not minimum <= count <= self.max_keys or len(node.values) != count:
                raise ValueError("BTree: occupancy or key/value count mismatch")
            if any(a >= b for a, b in zip(node.keys, node.keys[1:])):
                raise ValueError("BTree: keys must be strictly increasing")
            if node.keys and ((lower is not None and node.keys[0] <= lower)
                              or (upper is not None and node.keys[-1] >= upper)):
                raise ValueError("BTree: key outside subtree range")
            if node.leaf:
                if node.children:
                    raise ValueError("BTree: leaf has children")
                depths.add(depth)
                return count
            if len(node.children) != count + 1:
                raise ValueError("BTree: child count mismatch")
            bounds = [lower, *node.keys, upper]
            return count + sum(visit(child, depth + 1, bounds[i], bounds[i + 1])
                               for i, child in enumerate(node.children))

        total = visit(self.root, 0)
        if total != self._size or len(depths) != 1:
            raise ValueError("BTree: size mismatch or unequal leaf depths")
