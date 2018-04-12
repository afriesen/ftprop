# from copy import deepcopy
from enum import Enum, unique
from functools import partial
import numpy as np

import torch


def sign11(x):
    """take the sign of the input, and set sign(0) = -1, so that output \in {-1, +1} always"""
    return torch.sign(x).clamp(min=0) * 2 - 1


def hinge(z, t, margin=1.0, trunc_thresh=float('inf'), scale=1.0):
    """compute hinge loss for each input (z) w.r.t. each target (t)"""
    loss = ((-z * t.float() + margin) * scale).clamp(min=0, max=trunc_thresh)  # .mean(dim=0).sum()
    return loss


def dhinge_dz(z, t, margin=1.0, trunc_thresh=float('inf'), norm_by_size=True):
    """compute derivative of hinge loss w.r.t. input z"""
    tz = z * t
    dhdz = (torch.gt(tz, margin - trunc_thresh) * torch.le(tz, margin)).float() * -t
    if norm_by_size:
        dhdz = dhdz * (1.0 / tz.size()[0])
    return dhdz


def hingeL2(z, t, margin=1.0, trunc_thresh=float('inf')):
    loss = (margin - (z * t).float().clamp(min=trunc_thresh)).clamp(min=0)
    loss = loss * loss / 2
    return loss


def log_hinge(z, t, margin=1.0, trunc_thresh=float('inf'), scale=1.0):
    loss = (torch.log(1.0 + margin - (z * t.float()).clamp(min=-margin, max=margin)) * scale)
    return loss


def sigmoid(z, t, xscale=1.0, yscale=1.0):
    loss = torch.sigmoid(-(z * t).float() * xscale) * yscale
    return loss


def log_sigmoid(z, t, xscale=1.0, yscale=1.0):
    return torch.log(sigmoid(z, t, xscale, yscale))


def log_loss(z, t, trunc_thresh=float('inf')):
    loss = torch.log(1.0 + torch.exp(-z * t)).clamp(max=trunc_thresh)
    return loss


def square_loss(z, t, margin=1.0, scale=1.0, trunc_thresh=float('inf')):
    loss = (((margin - (z * t).clamp(max=1)) ** 2) * scale).clamp(max=trunc_thresh)
    return loss


def soft_hinge(z, t, xscale=1.0, yscale=1.0):
    loss = yscale * torch.tanh(-(z * t).float() * xscale) + 1
    return loss


def hinge11(z, t, margin=1.0, trunc_thresh=2):
    loss = (-z * t.float() + margin).clamp(min=0, max=trunc_thresh) - 1.0
    return loss


def is_step_module(module):
    mstr = str(module)
    return mstr[0:5] == 'Step(' or mstr[0:10] == 'Staircase(' or mstr[0:13] == 'OldStaircase('


@unique
class TPRule(Enum):
    # the different targetprop rules for estimating targets and updating weights
    WtHinge = 0
    WeightedPerceptron = 2
    Adaline = 3
    STE = 4
    SSTE = 5
    TruncWtHinge = 8
    LogWtHinge = 14
    TruncWtL2Hinge = 15
    TruncLogWtHinge = 16
    TruncWtPerceptron = 17
    Sigmoid = 18
    LogLoss = 19
    TruncLogLoss = 20
    SquareLoss = 21
    TruncSquareLoss = 22
    TruncWtHinge2 = 23
    Sigmoid2_2 = 24
    LogSigmoid = 25
    TruncWtHinge11 = 26
    SoftHinge = 27
    SoftHinge2 = 28
    Sigmoid3_2 = 29
    SSTEv2 = 30
    LeCunTanh = 31
    Ramp = 32

    @staticmethod
    def wt_hinge_backward(step_input, grad_output, target, is01):
        if target is None:
            target = -torch.sign(grad_output)
        assert False
        return dhinge_dz(step_input, target, margin=1), None

    @staticmethod
    def wt_perceptron_backward(step_input, grad_output, target, is01):
        assert not is01
        target = -torch.sign(grad_output) if target is None else target
        return dhinge_dz(step_input, target, margin=0), None

    @staticmethod
    def adaline_backward(step_input, grad_output, target, is01):
        assert not is01, 'adaline backward doesn''t support is01 yet'
        target = -torch.sign(grad_output) if target is None else target
        return torch.mul(torch.abs(target), step_input - target) * (1.0 / step_input.size()[0]), None

    @staticmethod
    def ste_backward(step_input, grad_output, target, is01):
        return grad_output, None

    @staticmethod
    def sste_backward(step_input, grad_output, target, is01, a=1):
        if is01:
            grad_input = grad_output * torch.ge(step_input, 0).float() * torch.le(step_input, a).float()
        else:
            grad_input = grad_output * torch.le(torch.abs(step_input), a).float()
        return grad_input, None

    @staticmethod
    def trunc_wt_hinge_backward(step_input, grad_output, target, is01):
        if target is None:
            target = -torch.sign(grad_output)
        assert not is01, 'is01 not supported'
        grad_input = dhinge_dz(step_input, target, margin=1, trunc_thresh=2)
        return grad_input, None

    @staticmethod
    def sigmoid_backward(step_input, grad_output, target, is01, xscale=2.0, yscale=1.0):
        assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = sigmoid(step_input, target, xscale=xscale, yscale=1.0)
        grad_input = z * (1 - z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def tanh_backward(step_input, grad_output, target, is01, xscale=1.0, yscale=1.0):
        # assert not is01
        if target is None:
            target = torch.sign(-grad_output)
        z = soft_hinge(step_input, target, xscale=xscale, yscale=1.0) - 1
        grad_input = (1 - z * z) * xscale * yscale * -target / grad_output.size(0)
        return grad_input, None

    @staticmethod
    def ramp_backward(step_input, grad_output, target, is01):
        if target is None:
            target = torch.sign(-grad_output)
        abs_input = torch.abs(step_input)
        if is01:
            # grad_input = grad_output * ((step_input <= 1).float() * (step_input >= 0).float() +
            #                             abs_input * (step_input > -1).float() * (step_input < 0).float() +
            #                             (2 - abs_input) * (step_input < 2).float() * (step_input > 1).float())
            # ramp01 = @(zt) (0 <= zt) .* (zt <= 1) + ...
            #                 (zt + 1) .* (-1 < zt) .* (zt < 0) + ...
            #                 (2 - abs(zt)) .* (1 < zt) .* (zt < 2);
            ramp_input = ((0 <= step_input).float() * (step_input <= 1).float() +
                          (step_input+1) * (-1 < step_input).float() * (step_input < 0).float() +
                          (2 - abs_input) * (1 < step_input).float() * (step_input < 2).float())
        else:
            # grad_input = grad_output * ((abs_input <= 1).float() +
            #                             (2 - abs_input) * (abs_input < 2).float() * (abs_input > 1).float())
            # ramp = @(zt) ((abs(zt) <= 1) + ...
            #                 abs(2 - zt) .* (zt < 2) .* (zt > 1) + ...
            #                 abs(zt + 2) .* (zt < -1) .* (zt > -2));
            ramp_input = ((abs_input <= 1).float() +
                          (2 - step_input).abs_() * (1 < step_input).float() * (step_input < 2).float() +
                          (2 + step_input).abs_() * (-2 < step_input).float() * (step_input < -1).float())
        grad_input = grad_output * ramp_input
        return grad_input, None


    @staticmethod
    def get_backward_func(targetprop_rule):
        if targetprop_rule == TPRule.WtHinge:  # gradient of hinge loss
            tp_grad_func = TPRule.wt_hinge_backward
        elif targetprop_rule == TPRule.WeightedPerceptron:  # gradient of perceptron criterion
            tp_grad_func = TPRule.wt_perceptron_backward
        elif targetprop_rule == TPRule.Adaline:  # adaline / delta-rule update
            tp_grad_func = TPRule.adaline_backward
        elif targetprop_rule == TPRule.STE:
            tp_grad_func = TPRule.ste_backward
        elif targetprop_rule == TPRule.SSTE:
            tp_grad_func = TPRule.sste_backward
        elif targetprop_rule == TPRule.TruncWtHinge:  # or targetprop_rule == TPRule.GreedyTruncWtHinge:
                                                            # weighted hinge using a truncated hinge loss
            tp_grad_func = TPRule.trunc_wt_hinge_backward
        elif targetprop_rule == TPRule.Sigmoid:
            tp_grad_func = TPRule.sigmoid_backward
        elif targetprop_rule == TPRule.Sigmoid3_2:
            tp_grad_func = partial(TPRule.sigmoid_backward, xscale=3.0, yscale=2.0)
        elif targetprop_rule == TPRule.Sigmoid2_2:
            tp_grad_func = partial(TPRule.sigmoid_backward, xscale=2.0, yscale=2.0)
        elif targetprop_rule == TPRule.SoftHinge:
            tp_grad_func = partial(TPRule.tanh_backward, xscale=1.0)
        elif targetprop_rule == TPRule.LeCunTanh:
            tp_grad_func = partial(TPRule.tanh_backward, xscale=(2.0/3.0), yscale=1.7519)
        elif targetprop_rule == TPRule.SSTEv2:
            a = np.sqrt(12.0) / 2
            tp_grad_func = partial(TPRule.sste_backward, a=a)
        elif targetprop_rule == TPRule.Ramp:
            tp_grad_func = TPRule.ramp_backward
        else:
            raise ValueError('specified targetprop rule ({}) has no backward function'.format(targetprop_rule))
        return tp_grad_func

    @staticmethod
    def get_loss_func(targetprop_rule):
        if targetprop_rule == TPRule.WtHinge:
            tp_loss_func = hinge
        elif targetprop_rule == TPRule.TruncWtHinge:  # or targetprop_rule == TPRule.GreedyTruncWtHinge:
            tp_loss_func = partial(hinge, trunc_thresh=2)
        elif targetprop_rule == TPRule.TruncWtHinge2:
            # tp_loss_func = partial(hinge, trunc_thresh=2, scale=0.5)
            tp_loss_func = partial(hinge, trunc_thresh=2, scale=2)
        elif targetprop_rule == TPRule.TruncWtHinge11:
            tp_loss_func = hinge11
        elif targetprop_rule == TPRule.TruncWtL2Hinge:
            tp_loss_func = partial(hingeL2, trunc_thresh=-1)
        elif targetprop_rule == TPRule.LogWtHinge:
            tp_loss_func = log_hinge
        elif targetprop_rule == TPRule.TruncLogWtHinge:
            # tp_loss_func = partial(log_hinge, trunc_thresh=2)
            # tp_loss_func = partial(log_hinge, trunc_thresh=2, scale=0.5)
            tp_loss_func = partial(log_hinge, trunc_thresh=2, scale=2)
        elif targetprop_rule == TPRule.TruncWtPerceptron:
            tp_loss_func = partial(hinge, margin=0, trunc_thresh=1)
        elif targetprop_rule == TPRule.Sigmoid:
            tp_loss_func = partial(sigmoid, xscale=2.0)
            # tp_loss_func = partial(sigmoid, xscale=1.0)
        elif targetprop_rule == TPRule.Sigmoid2_2:
            tp_loss_func = partial(sigmoid, xscale=2.0, yscale=2.0)
        elif targetprop_rule == TPRule.Sigmoid3_2:
            tp_loss_func = partial(sigmoid, xscale=3.0, yscale=2.0)
        elif targetprop_rule == TPRule.LogSigmoid:
            tp_loss_func = partial(log_sigmoid, xscale=2.0, yscale=1.0)
        elif targetprop_rule == TPRule.LogLoss:
            tp_loss_func = log_loss
        elif targetprop_rule == TPRule.TruncLogLoss:
            tp_loss_func = partial(log_loss, trunc_thresh=2)
        elif targetprop_rule == TPRule.SquareLoss:
            tp_loss_func = square_loss
        elif targetprop_rule == TPRule.TruncSquareLoss:
            tp_loss_func = partial(square_loss, trunc_thresh=4)
        elif targetprop_rule == TPRule.SoftHinge:
            tp_loss_func = soft_hinge
        elif targetprop_rule == TPRule.SoftHinge2:
            tp_loss_func = partial(soft_hinge, xscale=2.0)
        elif targetprop_rule == TPRule.LeCunTanh:
            tp_loss_func = partial(soft_hinge, xscale=(2.0 / 3.0), yscale=1.7519)
        else:
            raise ValueError('targetprop rule ({}) does not have an associated loss function'.format(targetprop_rule))
        return tp_loss_func
