"""
    GNN training utility methods
"""

import networkx as nx
import copy
import numpy as np
import torch
import random
from gnn_constants_hetero import *


def check_if_graph_is_connected(edge_index):
    if type(edge_index) == torch.Tensor:
        edge_index = edge_index.numpy()
    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))

    s.extend(t)
    nodes = list(set(s))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    edges = np.array(edge_index)

    edges = [(row[0].item(), row[1].item()) for row in edges.T]
    graph.add_edges_from(edges)
    
    return nx.is_connected(graph)


def split_train_test_for_class(curr_class_list, cross_val_constant):
    random.shuffle(curr_class_list)
    list_len = len(curr_class_list)
    # print(list_len)
    train_ratio = (cross_val_constant - 1) / cross_val_constant
    train_set_len = int(list_len * train_ratio)
    train_class_index_list = curr_class_list[:train_set_len]
    test_class_index_list = curr_class_list[train_set_len:]

    return train_class_index_list, test_class_index_list

def prepare_train_test_masks(num_classes,true_classes):
    class_lists = []
    for i in range(num_classes):
        class_lists.append([])
    for y_i, y in enumerate(true_classes):
        class_lists[y].append(y_i)
    # Keep class ratio
    train_index_list = []
    test_index_list = []
    for i in range(num_classes):
        curr_train_index_list, curr_test_index_list = split_train_test_for_class(class_lists[i], CROSS_VAL_CONSTANT)
        train_index_list = train_index_list + curr_train_index_list
        test_index_list = test_index_list + curr_test_index_list
    train_index_list.sort()
    test_index_list.sort()

    train_mask = [0] * len(true_classes)
    test_mask = [0] * len(true_classes)

    train_mask = [1 if i in train_index_list else train_mask[i] for i in range(len(train_mask))]
    test_mask = [1 if i in true_classes else test_mask[i] for i in range(len(test_mask))]

    return train_mask,test_mask

def prepare_train_test_masks_for_cross_val(num_classes,true_classes):
    # Find indexes for class elements
    class_lists = []
    for i in range(num_classes):
        class_lists.append([])
    for y_i, y in enumerate(true_classes):
        class_lists[y].append(y_i)
    for c in range(num_classes):
        random.shuffle(class_lists[c])

    # Keep class ratio
    train_index_list = []
    test_index_list = []
    for i in range(CROSS_VAL_CONSTANT):

        curr_fold_train_index_list = []
        curr_fold_test_index_list = []
        for c in range(num_classes):
            len_curr_class = len(class_lists[c])
            num_inputs_for_test = int(len_curr_class/CROSS_VAL_CONSTANT)
            start_index_for_test_inputs = 0 + i*num_inputs_for_test
            end_index_for_test_inputs = start_index_for_test_inputs + num_inputs_for_test
            curr_class_test_index_list = class_lists[c][start_index_for_test_inputs:end_index_for_test_inputs]
            curr_class_train_index_list = list(set(class_lists[c]) - set(curr_class_test_index_list))

            curr_fold_train_index_list = curr_fold_train_index_list + curr_class_train_index_list
            curr_fold_test_index_list = curr_fold_test_index_list + curr_class_test_index_list

        curr_fold_train_index_list.sort()
        curr_fold_test_index_list.sort()
        train_index_list.append(curr_fold_train_index_list)
        test_index_list.append(curr_fold_test_index_list)

    return train_index_list, test_index_list


def prepare_train_test_masks_for_cross_val_with_val(num_classes,true_classes):
    print("prepare_train_test_masks_for_cross_val_with_val")
    # Find indexes for class elements
    class_lists = []
    for i in range(num_classes):
        class_lists.append([])
    for y_i, y in enumerate(true_classes):
        class_lists[y].append(y_i)
    for c in range(num_classes):
        random.shuffle(class_lists[c])

    # Keep class ratio
    train_index_list = []
    val_index_list = []
    test_index_list = []
    for i in range(CROSS_VAL_CONSTANT):
        #print(i)

        curr_fold_train_index_list = []
        curr_fold_val_index_list = []
        curr_fold_test_index_list = []
        for c in range(num_classes):
            len_curr_class = len(class_lists[c])

            num_inputs_for_test = int(len_curr_class/CROSS_VAL_CONSTANT)
            start_index_for_test_inputs = 0 + i*num_inputs_for_test
            end_index_for_test_inputs = start_index_for_test_inputs + num_inputs_for_test
            curr_class_test_index_list = class_lists[c][start_index_for_test_inputs:end_index_for_test_inputs]

            if i < 9:
                num_inputs_for_val = num_inputs_for_test
                start_index_for_val_inputs = end_index_for_test_inputs
                end_index_for_val_inputs = end_index_for_test_inputs + num_inputs_for_val
                curr_class_val_index_list = class_lists[c][start_index_for_val_inputs:end_index_for_val_inputs]
            else:
                num_inputs_for_val = num_inputs_for_test
                start_index_for_val_inputs = start_index_for_test_inputs - num_inputs_for_val
                end_index_for_val_inputs = start_index_for_test_inputs
                curr_class_val_index_list = class_lists[c][start_index_for_val_inputs:end_index_for_val_inputs]

            curr_class_train_index_list = list((set(class_lists[c]) - set(curr_class_val_index_list)) - set(curr_class_test_index_list))

            curr_fold_train_index_list = curr_fold_train_index_list + curr_class_train_index_list
            curr_fold_val_index_list = curr_fold_val_index_list + curr_class_val_index_list
            curr_fold_test_index_list = curr_fold_test_index_list + curr_class_test_index_list

        curr_fold_train_index_list.sort()
        curr_fold_val_index_list.sort()
        curr_fold_test_index_list.sort()

        #print(len(curr_fold_train_index_list))
        #print(len(curr_fold_val_index_list))
        #print(len(curr_fold_test_index_list))
        #print(len(curr_fold_train_index_list)+len(curr_fold_val_index_list)+len(curr_fold_test_index_list))

        train_index_list.append(curr_fold_train_index_list)
        val_index_list.append(curr_fold_val_index_list)
        test_index_list.append(curr_fold_test_index_list)

    return train_index_list, val_index_list, test_index_list


def pass_data_iteratively(model, graphs, minibatch_size = 32):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

# Define Jaccard Similarity function for two binary vectors
def jaccard_binary(x,y):
    # Compare columns with 1 as value in either of the vectors
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

# Define Jaccard Similarity function for two sets
def jaccard_set(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
