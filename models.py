import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch import relu, sigmoid
from collections import OrderedDict
import torch.nn.modules.activation as activation


#######################################################################
class ConvNetInterpPredictionMyModule(nn.Module):
    def __init__(self, num_of_cnns, num_classes, weight_path=None):
        super(ConvNetInterpPredictionMyModule, self).__init__()
        
        self.linears = nn.ModuleList([nn.Sequential(
                                        nn.Conv1d(4,1,20),
                                        nn.BatchNorm1d(1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool1d(7,7),
                                        nn.Flatten(),
                                        nn.Linear(25,100),
                                        nn.BatchNorm1d(100,1e-05,0.1,True),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.3),
                                        nn.Linear(100,1),
                                        nn.BatchNorm1d(1,1e-05,0.1,True),
                                        nn.ReLU(),
                                        nn.Dropout(0.3)) for i in range(num_of_cnns)])
        # print('model')
        # # print(self.linear)
        # print('\n')
        print('num_of_cnns {}'.format(num_of_cnns))
        self.final = nn.Linear(num_of_cnns, num_classes) #was 10
        
        if weight_path :
            self.load_weights(weight_path)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        outs = []
        for l in self.linears:
            # print('model')
            # print(l)
            # print('\n')
            outs.append(l(x))
        # print('out 0 shape {}'.format(outs[0].shape))
        # out = torch.cat(tuple(outs), 1)
        out = torch.cat(outs, 1)

        # print('out 5 shape {}'.format(out.shape))
        out = self.final(out)
        # print('out 10 shape {}'.format(out.shape))

        return out
    
    def load_weights(self, weight_path):
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)
       
    
# #######################################################################
# class motifCNNMyModule(nn.Module):
#     def __init__(self, num_of_cnns, original_model, num_classes):
#         super(motifCNNMyModule, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.linears = original_model.linears
#
#         self.final = original_model.final
#
#     def forward(self, input):
#         acts = []
#         outs = []
#         for l in self.linears:
#             acts.append(l[:3](input))
#             outs.append(l(input))
#
#         #we save the activations of the first layer
#         activations = torch.cat(tuple(acts), 1)
#
#         cnn_outputs = torch.cat(tuple(outs), 1)
#
#         return cnn_outputs, activations
#
