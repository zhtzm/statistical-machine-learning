import numpy as np


class MaxHeap(object):
    def __init__(self):
        self.array = [(0,)]
        self.size = 0

    def __len__(self):
        return self.size

    def pop(self):
        self.array[1] = self.array.pop()
        self.size -= 1
        self.heapify(1)

    def push(self, item: tuple):
        self.array.append(item)
        self.size += 1

        index = self.size
        while index > 1:
            parent = index // 2
            if self.array[index][-1] > self.array[parent][-1]:
                self.array[index], self.array[parent] = \
                    self.array[parent], self.array[index]
                index = parent
            else:
                break

    def heapify(self, root):
        if len(self) == 0 or len(self) == 1:
            return

        left = 2 * root
        right = left + 1
        max_n = root

        if left <= len(self) and self.array[left][-1] > self.array[max_n][-1]:
            max_n = left
        if right <= len(self) and self.array[right][-1] > self.array[max_n][-1]:
            max_n = left

        if max_n is not root:
            self.array[max_n], self.array[root] = self.array[root], self.array[max_n]
            self.heapify(max_n)

    def peak(self):
        return self.array[1]
