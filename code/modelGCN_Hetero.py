import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv
from gnn_constants_hetero import *

class modelGCN_Hetero(torch.nn.Module):

    def __init__(self, num_gcn_layers, num_lin_layers, input_dim, hidden_gcn_dim, output_dim, dropout_rate):
        super(modelGCN_Hetero, self).__init__()
        torch.manual_seed(783)

        self.num_gcn_layers = num_gcn_layers
        self.num_lin_layers = num_lin_layers
        self.input_dim = input_dim
        self.hidden_gcn_dim = hidden_gcn_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


        # Initialize GCN layers
        self.conv = torch.nn.ModuleList()
        first_conv = GCNConv(input_dim, hidden_gcn_dim, add_self_loops=False)
        self.conv.append(first_conv)
        for l in range(num_gcn_layers-1):
            curr_conv = GCNConv(hidden_gcn_dim, hidden_gcn_dim, add_self_loops=False)
            self.conv.append(curr_conv)
        linear_input_dim = hidden_gcn_dim

        # Initialize Linear layer
        self.out = torch.nn.ModuleList()
        for l in range(num_lin_layers-1):
            if LIN_MODEL_HALF == 1:
                linear_output_dim = int(linear_input_dim/2)
            else:
                linear_output_dim = linear_input_dim
            curr_lin = Linear(linear_input_dim, linear_output_dim)
            linear_input_dim = linear_output_dim
            self.out.append(curr_lin)
        self.out.append(Linear(linear_input_dim, output_dim))


    def forward(self, x, edge_index):

        # Message Passing Layers
        for l in range(self.num_gcn_layers):
            x = self.conv[l](x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Output layer
        for l in range(self.num_lin_layers):
            x = F.softmax(self.out[l](x), dim=1)

        return x