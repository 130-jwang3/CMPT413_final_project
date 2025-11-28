from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math

# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class Generator(nn.Module):
    def __init__(self, win_size, latent_dim, input_c=1, dropout=0.2):
        super(Generator, self).__init__()
        self.win_size = win_size
        self.input_c = input_c
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, win_size * input_c),
            nn.Tanh(),                     # scale to [-1,1]; adjust if you prefer [0,1]
        )

    def forward(self, z):                  # z: (B, latent_dim)
        x = self.proj(z)                   # (B, win_size*input_c)
        x = x.view(z.size(0), self.input_c, self.win_size)  # (B, C, W)
        return x


class Discriminator(nn.Module):
    def __init__(self, win_size, input_c=1, dropout=0.2):
        super(Discriminator, self).__init__()
        self.win_size = win_size
        self.input_c = input_c

        self.feature = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(input_c, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Infer flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, input_c, win_size)
            f = self.feature(dummy)
            flat = f.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)  # logit
        )

    def forward(self, x):                  # x: (B, C, W) or (B, W) or (B, W, C)
        # Coerce to (B, C, W)
        if x.dim() == 2:                   # (B, W) -> (B,1,W)
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # If (B, W, C) -> (B, C, W)
            if x.shape[1] == self.win_size and x.shape[2] == self.input_c:
                x = x.transpose(1, 2).contiguous()
            # If already (B, C, W), leave it

        f = self.feature(x)
        return self.classifier(f)          # (B, 1)


## LSTM_AD Model
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super(LSTM_AD, self).__init__()
        self.name = 'LSTM_AD'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden, batch_first=False)  # Input size = feats, seq_len first
        self.lstm2 = nn.LSTM(self.n_hidden, self.n_feats, batch_first=False)  # Output size = n_feats
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        # x shape: (seq_len, batch, input_size) e.g., (600, 32, 38)
        batch_size = x.size(1)
        hidden = (torch.zeros(1, batch_size, self.n_hidden).to(x.device),
                  torch.zeros(1, batch_size, self.n_hidden).to(x.device))
        hidden2 = (torch.zeros(1, batch_size, self.n_feats).to(x.device),
                   torch.zeros(1, batch_size, self.n_feats).to(x.device))

        # Process the entire sequence
        out, (hidden, _) = self.lstm(x, hidden)  # out: (seq_len, batch, n_hidden)
        out, (hidden2, _) = self.lstm2(out, hidden2)  # out: (seq_len, batch, n_feats)
        out = self.fcn(out)  # (seq_len, batch, n_feats)
        return out  # Return the full sequence output


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GAT(nn.Module):
    def __init__(self, num_node_features, inter_dim, num_classes):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels=num_node_features, out_channels=inter_dim, heads=2)
        self.conv2 = GATConv(in_channels=2 * inter_dim, out_channels=inter_dim, heads=2)
        self.fc = nn.Linear(inter_dim * 2, 1)  # Output layer for regression

    def forward(self, x_embedding, edge_index):
        x, edge_index = x_embedding, edge_index
        x1 = self.conv1(x, edge_index)
        x1 = F.dropout(x1, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = F.dropout(x2, training=self.training)

        x2 = self.fc(x2)

        return x2


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0):
        super(GNNLayer, self).__init__()

        self.gnn = GAT(in_channel, num_classes=out_channel, inter_dim=inter_dim)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        out = self.gnn(x, edge_index)
        out = self.bn(out)
        out = self.relu(out)
        return out


class GDN(nn.Module):
    def __init__(self,
                 edge_index_sets,
                 node_num,
                 win_size=10,
                 out_layer_num=1,
                 out_layer_inter_dim=256, ):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        edge_set_num = len(edge_index_sets)
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.out_layer_inter_dim = out_layer_inter_dim

        self.gnn_layers = GNNLayer(in_channel=win_size, out_channel=out_layer_num, inter_dim=out_layer_inter_dim)

        self.bn_outlayer_in = nn.BatchNorm1d(out_layer_inter_dim)
        self.dp = nn.Dropout(0.2)

    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        batch_num, all_feature, node_num = x.shape
        x = x.view(-1, all_feature).contiguous()  # x={Tensor:(bn*node_num,all_feature)}

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            gcn_out = self.gnn_layers(x, batch_edge_index)

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)

        out = x.view(-1, 1, node_num)

        return out