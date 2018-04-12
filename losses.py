import torch
import torch.nn.functional as F
from torch.autograd import Variable


def convert_to_1hot(target, noutputs, make_11=True):
    if target.is_cuda:
        target_1hot = torch.cuda.CharTensor(target.size()[0], noutputs)
    else:
        target_1hot = torch.CharTensor(target.size()[0], noutputs)
    target_1hot.zero_()
    target_1hot.scatter_(1, target.unsqueeze(1), 1)
    target_1hot = target_1hot.type(target.type())
    if make_11:
        target_1hot = target_1hot * 2 - 1
    return target_1hot


def multiclass_hinge_loss(input_, target):
    """ compute hinge loss: max(0, 1 - input * target) """
    target_1hot = convert_to_1hot(target.data, input_.size()[1], make_11=True).float()
    if type(input_) is Variable:
        target_1hot = Variable(target_1hot)
    loss = (-target_1hot * input_.float() + 1.0).clamp(min=0).sum(dim=1).mean(dim=0)  # max(0, 1-z*t)
    return loss


def multiclass_squared_hinge_loss(input_, target):
    """ compute squared hinge loss: max(0, 1 - input * target)^2 """
    target_1hot = convert_to_1hot(target.data, input_.size()[1], make_11=True).float()
    if type(input_) is Variable:
        target_1hot = Variable(target_1hot)
    loss = (-target_1hot * input_.float() + 1.0).clamp(min=0).pow(2).sum(dim=1).mean(dim=0)  # max(0, 1-z*t)^2
    return loss


def multiclass_truncated_hinge_loss(input_, target, thresh=2):
    """ compute truncated hinge loss: min(thresh, max(0, 1 - input * target)) """
    target_1hot = convert_to_1hot(target.data, input_.size()[1], make_11=True).float()
    if type(input_) is Variable:
        target_1hot = Variable(target_1hot)
    loss = (-target_1hot * input_.float() + 1.0).clamp(min=0, max=thresh).sum(dim=1).mean(dim=0)
    return loss


def multiclass_hinge_loss_softmax(input_, target):
    """ compute hinge loss after performing a (log) softmax on the input: max(0, 1 - log(softmax(input)) * target) """
    x = F.log_softmax(input_)
    return multiclass_hinge_loss(x, target)


def multiclass_trunc_hinge_loss_softmax(input_, target, thresh=2):
    """
    compute truncated hinge loss after performing a (log) softmax on the input:
        min(thresh, max(0, 1 - log(softmax(input)) * target))
    """
    x = F.log_softmax(input_)
    return multiclass_truncated_hinge_loss(x, target, thresh=thresh)
