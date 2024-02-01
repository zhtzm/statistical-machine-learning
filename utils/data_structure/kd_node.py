class KDNode(object):
    def __init__(self, parent, value, axis):
        self.data = value[: -1]
        self.label = value[-1]
        self.parent = parent
        self.axis = axis
        self.left = None
        self.right = None
