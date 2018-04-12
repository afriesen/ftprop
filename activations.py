from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
import targetprop as tp


class StepF(Function):
    """
    A step function that returns values in {-1, 1} and uses targetprop to
    update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.WtHinge, make01=False, scale_by_grad_out=False):
        super(StepF, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out
        self.target = None
        self.saved_grad_out = None
        if self.tp_rule in [tp.TPRule.SSTE, tp.TPRule.STE, tp.TPRule.Ramp]:
            assert self.scale_by_grad_out, 'scale_by_grad_out is required for {}'.format(self.tp_rule.name)
            self.scale_by_grad_out = False # require this set as input, but
                                           # don't actually use it since these
                                           # methods handle it internally

    def forward(self, input_):
        self.save_for_backward(input_)
        # output = torch.sign(input_)  # output \in {-1, 0, +1}
        output = tp.sign11(input_)  # output \in {-1, +1}
        if self.make01:
            output.clamp_(min=0)  # output \in {0, 1}
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        grad_input = None
        if self.needs_input_grad[0]:
            # compute targets = neg. sign of output grad, where t \in {-1, 0, 1} (t=0 means ignore this unit)
            go = grad_output if self.saved_grad_out is None else self.saved_grad_out
            tp_grad_func = tp.TPRule.get_backward_func(self.tp_rule)
            grad_input, self.target = tp_grad_func(input_, go, self.target, self.make01)
            if self.scale_by_grad_out:
                grad_input = grad_input * go.size()[0] * go.abs()  # remove batch-size scaling
        return grad_input


class Step(nn.Module):
    """
    Module wrapper for a step function (StepF).
    """

    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, make01=False, scale_by_grad_out=False):
        super(Step, self).__init__()
        self.tp_rule = targetprop_rule
        self.make01 = make01
        self.scale_by_grad_out = scale_by_grad_out

    def __repr__(self):
        s = '{name}(a={a}, b={b}, tp={tp})'
        a = 0 if self.make01 else -1
        return s.format(name=self.__class__.__name__, a=a, b=1, tp=self.tp_rule)

    def forward(self, x):
        y = StepF(targetprop_rule=self.tp_rule, make01=self.make01,
                  scale_by_grad_out=self.scale_by_grad_out)(x)
        return y


class StaircaseF(Function):
    """
    A staircase function (implicit sum of step functions) that uses targetprop
    to update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge,
                 nsteps=5, margin=1, trunc_thresh=2, scale_by_grad_out=False):
        super(StaircaseF, self).__init__()
        assert nsteps >= 1, 'must be at least one step in the staircase'
        assert nsteps < 255, 'saving as byte will fail here'
        self.tp_rule = targetprop_rule
        # self.tp_grad_func = TargetpropRule.getBackwardFunc(self.tp_rule)
        self.nsteps, self.m, self.q = nsteps, margin, trunc_thresh
        self.scale_by_grad_out = scale_by_grad_out

    def forward(self, x):
        self.save_for_backward(x)
        z = x * (self.nsteps - 1)                 # rescale values from [0, 1] to [0, nsteps-1]
        z.ceil_().clamp_(min=0, max=self.nsteps)  # quantize and then clip any values outside range [0, nsteps]
        z *= (1.0 / self.nsteps)                  # rescale into [0, 1]
        return z

    def backward(self, grad_output):
        x, = self.saved_tensors
        grad_input = None
        if self.needs_input_grad[0]:
            if self.tp_rule == tp.TPRule.SSTE:
                grad_input = grad_output * (torch.le(x, 1) * torch.ge(x, 0)).float()
                assert self.scale_by_grad_out, "SSTE requires scale_by_grad_out"

            elif self.tp_rule == tp.TPRule.TruncWtHinge:
                target = -torch.sign(grad_output)
                z = x * (self.nsteps - 1)
                m = self.m * (self.nsteps - 1)
                q = self.q * (self.nsteps - 1)
                pos_target = ((z + q - m).clamp_(min=0, max=self.nsteps).ceil_() -
                              (z - m).clamp_(min=0, max=self.nsteps).ceil_())
                neg_target = ((z + m).clamp_(min=0, max=self.nsteps).ceil_() -
                              (z + m - q).clamp_(min=0, max=self.nsteps).ceil_())

                grad_input = -target * pos_target * (target > 0).float() + neg_target * (target < 0).float()
                grad_input = grad_input * (1. / self.nsteps)

                if self.scale_by_grad_out:
                    grad_input = grad_input * grad_output.abs()
                else:
                    grad_input = grad_input * (1.0 / grad_output.size(0))  # normalize by batch size

            elif self.tp_rule == tp.TPRule.SoftHinge:
                target = torch.sign(-grad_output)

                # compute f(x) = (tanh(2*x-1)+1)/2;   f'(x) = (1 - tanh(2x-1)^2)*2/2 = (1-tanh(2x-1))
                z = tp.soft_hinge(2.0 * x - 1, target, xscale=1.0) - 1
                grad_input = (1 - z * z) * -target

                # # # compute f(x) = tanh(x-0.5) + 0.5;    f'(x) = (1 - tanh(x-0.5)^2)
                # # z = tp.tanh(x - 0.5, target, xscale=1.0) - 1
                # # grad_input = (1 - z * z) * -target

                # # # compute f(x) = 0.25*tanh(4*(x-0.5))+0.5;   f'(x) = (1 - tanh(4*x-2)^2)
                # # z = tp.tanh(4.0*x-2.0, target, xscale=1.0) - 1
                # # grad_input = (1 - z * z) * -target

                if self.scale_by_grad_out:
                    grad_input = grad_input * grad_output.abs()
                else:
                    grad_input = grad_input * (1.0 / grad_output.size(0))  # normalize by batch size
            else:
                raise ValueError('only SSTE and TruncWtHinge are supported for staircase activation')
        return grad_input


class Staircase(nn.Module):
    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, nsteps=5, margin=1, trunc_thresh=2, a=0, b=1,
                 scale_by_grad_out=False):
        super(Staircase, self).__init__()
        self.tp_rule = targetprop_rule
        self.nsteps = nsteps
        self.margin = margin
        self.trunc_thresh = trunc_thresh
        self.a, self.b = a, b
        self.scale_by_grad_out = scale_by_grad_out

    def forward(self, x):
        x = (x - self.a) / (self.b - self.a)  # shift and rescale x \in [a, b] to be in [0, 1]
        y = StaircaseF(targetprop_rule=self.tp_rule, nsteps=self.nsteps,
                       margin=self.margin, trunc_thresh=self.trunc_thresh,
                       scale_by_grad_out=self.scale_by_grad_out)(x)
        y = y * (self.b - self.a) + self.a   # shift and rescale y \in [0, 1] to be in [a, b]
        return y

    def __repr__(self):
        s = '{}(steps={}, margin={}, thresh={})'
        return s.format(self.__class__.__name__, self.nsteps, self.margin, self.trunc_thresh)


class OldStaircase(nn.Module):
    """
    Old, inefficient staircase function constructed as a sum of step functions
    that each uses targetprop to update upstream weights in the network.
    """

    def __init__(self, targetprop_rule=tp.TPRule.TruncWtHinge, nsteps=5, margin=1, trunc_thresh=2, a=0, b=1,
                 scale_by_grad_out=False):
        super(OldStaircase, self).__init__()
        self.tp_rule = targetprop_rule
        self.nsteps = nsteps
        self.margin = margin
        self.trunc_thresh = trunc_thresh
        self.a, self.b = a, b
        self.scale_by_grad_out = scale_by_grad_out
        self.step_func = partial(StepF, targetprop_rule=self.tp_rule, make01=True,
                                 scale_by_grad_out=self.scale_by_grad_out)

    def forward(self, x):
        x, y = (x - self.a), 0.0
        delta = (self.b - self.a) / float(self.nsteps - 1)
        for i in range(0, self.nsteps):
            y = y + self.step_func()(x - i * delta)
        return y * (1.0 / self.nsteps) * (self.b - self.a) + self.a

    def __repr__(self):
        s = '{}(a={}, b={}, steps={}, margin={}, thresh={}, tp={})'
        return s.format(self.__class__.__name__, self.a, self.b, self.nsteps,
                        self.margin, self.trunc_thresh, self.tp_rule)


class ThresholdReLU(nn.Module):
    def __init__(self, max_val=1., slope=1.):
        super(ThresholdReLU, self).__init__()
        self.max_val = max_val
        self.slope = slope

    def forward(self, x):
        return F.relu(x * self.slope if self.slope != 1 else x).clamp(max=self.max_val)


class CAbs(nn.Module):
    def __init__(self):
        super(CAbs, self).__init__()

    def forward(self, x):
        return x.abs().clamp(max=1.0)


def step(input, targetprop_rule=tp.TPRule.TruncWtHinge, scale_by_grad_out=False):
    return StepF(targetprop_rule=targetprop_rule, make01=False,
                 scale_by_grad_out=scale_by_grad_out)(input)


def step01(input, targetprop_rule=tp.TPRule.TruncWtHinge, scale_by_grad_out=False):
    return StepF(targetprop_rule=targetprop_rule, make01=True,
                 scale_by_grad_out=scale_by_grad_out)(input)


def thresholdrelu(input, max_val=1., slope=1.):
    return F.relu(input * slope if slope != 1 else input).clamp(max=max_val)


def hardsigmoid(input):
    return thresholdrelu(input + 1, max_val=1, slope=0.5)


def staircase(input, targetprop_rule=tp.TPRule.TruncWtHinge,
              nsteps=5, margin=1, trunc_thresh=2):
    return StaircaseF(targetprop_rule=targetprop_rule,
                      nsteps=nsteps, margin=margin,
                      trunc_thresh=trunc_thresh)(input)


def staircase_old(z, targetprop_rule=tp.TPRule.TruncWtHinge, a=0, b=1,
                  nsteps=5, scale_by_grad_out=False):
    y = 0.0
    z = z - a
    delta = (b - a) / float(nsteps - 1)
    for i in range(0, nsteps):
        y = y + step01(z - i * delta, targetprop_rule=targetprop_rule,
                       scale_by_grad_out=scale_by_grad_out)
    return y * (1. / nsteps)
