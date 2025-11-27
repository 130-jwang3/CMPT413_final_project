import torch


def generate_fc_edge_index(num_nodes):
    node_index = torch.arange(num_nodes)

    edge_index = torch.cartesian_prod(node_index, node_index)

    edge_index = edge_index[edge_index[:, 0] != edge_index[:, 1]]

    fc_edge_index = edge_index.t()

    swapped_tensor = fc_edge_index[[1, 0], :]

    return swapped_tensor

if __name__ == '__main__':
    print(generate_fc_edge_index(6))
