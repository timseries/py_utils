class _BPlusLeaf(_BNode):
	__slots__ = ["tree", "contents", "data", "next"]

	def __init__(self, tree, contents=None, data=None, next=None):
		self.tree = tree
		self.contents = contents or []
		self.data = data or []
		self.next = next
		assert len(self.contents) == len(self.data), "one data per key"

	def insert(self, index, key, data, ancestors):
		self.contents.insert(index, key)
		self.data.insert(index, data)

		if len(self.contents) > self.tree.order:
			self.shrink(ancestors)

	def lateral(self, parent, parent_index, dest, dest_index):
		if parent_index > dest_index:
			dest.contents.append(self.contents.pop(0))
			dest.data.append(self.data.pop(0))
			parent.contents[dest_index] = self.contents[0]
		else:
			dest.contents.insert(0, self.contents.pop())
			dest.data.insert(0, self.data.pop())
			parent.contents[parent_index] = dest.contents[0]

	def split(self):
		center = len(self.contents) // 2
		median = self.contents[center - 1]
		sibling = type(self)(
			self.tree,
			self.contents[center:],
			self.data[center:],
			self.next)
		self.contents = self.contents[:center]
		self.data = self.data[:center]
		self.next = sibling
		return sibling, sibling.contents[0]

	def remove(self, index, ancestors):
		minimum = self.tree.order // 2
		if index >= len(self.contents):
			self, index = self.next, 0

		key = self.contents[index]

		# if any leaf that could accept the key can do so
		# without any rebalancing necessary, then go that route
		current = self
		while current is not None and current.contents[0] == key:
			if len(current.contents) > minimum:
				if current.contents[0] == key:
					index = 0
				else:
					index = bisect.bisect_left(current.contents, key)
				current.contents.pop(index)
				current.data.pop(index)
				return
			current = current.next

		self.grow(ancestors)

	def grow(self, ancestors):
		minimum = self.tree.order // 2
		parent, parent_index = ancestors.pop()
		left_sib = right_sib = None

		# try borrowing from a neighbor - try right first
		if parent_index + 1 < len(parent.children):
			right_sib = parent.children[parent_index + 1]
			if len(right_sib.contents) > minimum:
				right_sib.lateral(parent, parent_index + 1, self, parent_index)
				return

		# fallback to left
		if parent_index:
			left_sib = parent.children[parent_index - 1]
			if len(left_sib.contents) > minimum:
				left_sib.lateral(parent, parent_index - 1, self, parent_index)
				return

		# join with a neighbor - try left first
		if left_sib:
			left_sib.contents.extend(self.contents)
			left_sib.data.extend(self.data)
			parent.remove(parent_index - 1, ancestors)
			return

		# fallback to right
		self.contents.extend(right_sib.contents)
		self.data.extend(right_sib.data)
		parent.remove(parent_index, ancestors)
