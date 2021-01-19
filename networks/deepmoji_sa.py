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



class FEDA_classifier(nn.Module):
    def __init__(self, args):
        super(FEDA_classifier, self).__init__()
        self.args = args
        self.emb_size = self.args.emb_size
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        self.LeakyReLU = nn.LeakyReLU()

        # adv level
        self.adv_level = self.args.adv_level

        # first fully connected layer
        self.fc1_s = nn.Linear(self.emb_size, self.hidden_size)
        self.fc1_p1 = nn.Linear(self.emb_size, self.hidden_size)
        self.fc1_p2 = nn.Linear(self.emb_size, self.hidden_size)
        # second fully connected layer
        self.fc2_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_p1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_p2 = nn.Linear(self.hidden_size, self.hidden_size)
        # output layer
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes) 

    def forward(self, input, private_info):
        if self.args.original_FEDA:
            # Performaing original FEDA
            # Process each channel saperately and add outputs together before the last layer. 
            # Shared 
            out_s = self.fc1_s(input)
            out_s = self.LeakyReLU(out_s)                        
            out_s = self.fc2_s(out_s)

            # Private 1
            out_p1 = self.fc1_p1(input)
            out_p1 = self.LeakyReLU(out_p1)
            out_p1 = self.fc2_p1(out_p1)

            # Private 2 
            out_p2 = self.fc1_p2(input)
            out_p2 = self.LeakyReLU(out_p2)
            out_p2 = self.fc2_p2(out_p2)

            # selecting channels according to private info
            # removing a certain channel by times 0 
            out_p1 = out_p1 * (private_info.float().reshape(-1,1))
            out_p2 = out_p2 * (((private_info-1)**2).float().reshape(-1,1))
            # Merging
            out = torch.add(torch.add(out_s, out_p1), out_p2) # Batch_size * hidden_size

            # Output
            out = self.fc3(out)
            return out
        else:
            # FEDA-Performing correction at each layer
            # Use private information at each layer

            # First layer
            out_s = self.fc1_s(input)
            out_p1 = self.fc1_p1(input)
            out_p2 = self.fc1_p2(input)
            # correction
            out_p1 = out_p1 * (private_info.float().reshape(-1,1))
            out_p2 = out_p2 * (((private_info-1)**2).float().reshape(-1,1))
            # adding
            corrected_out_1 = torch.add(torch.add(out_s, out_p1), out_p2)
            
            # Activation function
            corrected_out_1 = self.LeakyReLU(corrected_out_1)

            # Second layer
            out_s = self.fc2_s(corrected_out_1)
            out_p1 = self.fc2_p1(corrected_out_1)
            out_p2 = self.fc2_p2(corrected_out_1)
            # correction
            out_p1 = out_p1 * (private_info.float().reshape(-1,1))
            out_p2 = out_p2 * (((private_info-1)**2).float().reshape(-1,1))
            # adding
            corrected_out_2 = torch.add(torch.add(out_s, out_p1), out_p2)
            
            # Output
            out = self.fc3(corrected_out_2)
            return out

    def shared_pred(self, input):
        """
        Input: vectors
        Output:
            predictions
            shared representations
        """
        out_s = self.fc1_s(input)
        out_s = self.LeakyReLU(out_s)                        
        out_s = self.fc2_s(out_s) 
        # Output
        out = self.fc3(out)
        return out

    def hidden(self, input):
        """
        Input: vectors
        Output:
            predictions
            shared representations
        """
        assert self.adv_level in set([0, -1, -2])
        out_s = self.fc1_s(input)
        if self.adv_level == -2:
            return out_s
        else:
            out_s = self.LeakyReLU(out_s)
            out_s = self.fc2_s(out_s)
            if self.adv_level == -1:
                return out_s
            else:
                # Output
                out = self.fc3(out)
                return out