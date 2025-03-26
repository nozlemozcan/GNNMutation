import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from gnn_constants_hetero import *

class modelGAT_Hetero(torch.nn.Module):

    def __init__(self, num_gat_layers, num_lin_layers, input_dim, hidden_gat_dim, num_gat_heads, output_dim, dropout_rate):
        super(modelGAT_Hetero, self).__init__()
        torch.manual_seed(783)

        self.num_gat_layers = num_gat_layers
        self.num_lin_layers = num_lin_layers
        self.input_dim = input_dim
        self.hidden_gat_dim = hidden_gat_dim
        self.num_gat_heads = num_gat_heads
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


        # Initialize GAT layers
        self.conv = torch.nn.ModuleList()
        if num_gat_layers == 1:
            first_conv = GATv2Conv(input_dim, output_dim, heads=1, add_self_loops=False)
            self.conv.append(first_conv)
        else:
            first_conv = GATv2Conv(input_dim, hidden_gat_dim, heads=num_gat_heads, add_self_loops=False)
            self.conv.append(first_conv)
            for l in range(num_gat_layers-2):
                curr_conv = GATv2Conv(hidden_gat_dim*num_gat_heads, hidden_gat_dim, heads=num_gat_heads, add_self_loops=False)
                self.conv.append(curr_conv)
            last_conv = GATv2Conv(hidden_gat_dim*num_gat_heads, hidden_gat_dim, heads=1, add_self_loops=False)
            self.conv.append(last_conv)

        linear_input_dim = hidden_gat_dim

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
        for l in range(self.num_gat_layers):
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.conv[l](x, edge_index)
            if l < self.num_gat_layers-1:
                x = F.elu(x)

        # Output layer
        for l in range(self.num_lin_layers):
            x = F.softmax(self.out[l](x), dim=1)
        return x