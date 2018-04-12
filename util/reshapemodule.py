from torch.nn import Module


class ReshapeBatch(Module):
    """ A simple layer that reshapes its input and outputs the reshaped tensor """
    def __init__(self, *args):
        super(ReshapeBatch, self).__init__()
        self.args = args

    def forward(self, x):
        return x.view(x.size(0), *self.args)

    def __repr__(self):
        s = '{}({})'
        return s.format(self.__class__.__name__, self.args)
