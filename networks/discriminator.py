
import torch
from torch import nn as nn


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, args, input_size, num_classes):
        super(Discriminator, self).__init__()

        self.GR = False
        self.grad_rev = GradientReversal(args.LAMBDA)
        self.fc1 = nn.Linear(input_size, args.adv_units)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(args.adv_units, args.adv_units)
        self.fc3 = nn.Linear(args.adv_units, num_classes)

    def forward(self, input):
        if self.GR:
            input = self.grad_rev(input)
        out = self.fc1(input)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def hidden_representation(self, input):
        if self.GR:
            input = self.grad_rev(input)
        out = self.fc1(input)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        # Return the hidden representation from the second last layer
        return out
    
    def Change_GradientReversal(self, State=True):
        self.GR = State

    def get_weights(self):
        # return coef as numpy array
        dense_parameter = {name:param for name, param in self.fc3.named_parameters()}
        # get coef and covert to numpy
        # return dense_parameter["weight"].cpu().numpy()
        return dense_parameter["weight"].detach().cpu().numpy()

class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)