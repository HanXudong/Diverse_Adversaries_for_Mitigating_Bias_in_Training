import torch
import torch.optim as optim
import torch.nn as nn
from networks.discriminator import Discriminator

class DeepMojiModel(nn.Module):
    def __init__(self, args):
        super(DeepMojiModel, self).__init__()
        self.args = args
        self.emb_size = self.args.emb_size
        self.hidden_size = self.args.hidden_size
        self.num_classes = self.args.num_classes
        self.adv_level = self.args.adv_level

        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.AF = nn.Tanh()
        try:
            if args.AF == "relu":
                self.AF = self.ReLU
            elif args.AF == "tanh":
                self.AF = self.tanh
        except:
            pass
        self.dense1 = nn.Linear(self.emb_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input):
        out = self.dense1(input)
        out = self.AF(out)
        out = self.dense2(out)
        out = self.tanh(out)
        out = self.dense3(out)
        return out
    
    def hidden(self, input):
        assert self.adv_level in set([0, -1, -2])
        out = self.dense1(input)
        out = self.AF(out)
        if self.adv_level == -2:
            return out
        else:
            out = self.dense2(out)
            out = self.tanh(out)
            if self.adv_level == -1:
                return out
            else:
                out = self.dense3(out)
                return out