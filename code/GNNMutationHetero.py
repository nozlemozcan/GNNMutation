from dataset import load_dataset_hetero, load_dataset_hetero_for_simulation, load_dataset_hetero_for_leave1out, load_dataset_hetero_for_leave1out_test2
from gnn_training_utils import check_if_graph_is_connected, prepare_train_test_masks_for_cross_val, prepare_train_test_masks_for_cross_val_with_val
from gnn_constants_hetero import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,matthews_corrcoef,precision_score,recall_score,auc, precision_recall_curve
from torch_geometric.nn import to_hetero
import copy
import datetime

from modelGCN_Hetero import modelGCN_Hetero
from modelGAT_Hetero import modelGAT_Hetero


class GNNMutationHetero(object):

    def __init__(self,*args):
        self.init_crossval(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10])

    def init_crossval(self, model_selection=MODEL_GCN, file_prefix="general", location=None, features=None, similarity=None, ppi=None, pg=None, cutoff=950, undirected=False, early_stopping=False, model_name="") -> None:

        print("init_crossval")
        self.model_selection = model_selection
        self.file_prefix = file_prefix
        self.location = location
        self.features = features
        self.similarity = similarity
        self.ppi = ppi
        self.pg = pg
        self.model_name = model_name

        self.dataset = None
        self.model_status = None
        self.model = None
        self.gene_names = None
        self.accuracy = None
        self.confusion_matrix = None
        self.test_loss = None
        self.early_stopping = early_stopping
        self.add_selfloops = ADD_SELFLOOPS_FLAG
        self.add_sim = ADD_SIM_EDGES

        # Flags for internal use (hidden from user)
        self._explainer_run = False

        # Check if graph exists
        if ppi == None:
            print("No ppi file!")
            return None

        if pg == None:
            print("No pg file!")
            return None

        dataset, gene_names, patients, true_classes, num_features, num_classes = load_dataset_hetero(self.features, self.similarity, self.ppi, self.pg, True, cutoff, undirected, self.add_selfloops, self.add_sim, self.file_prefix)

        self.dataset = dataset
        self.gene_names = gene_names
        self.patients = patients
        self.true_classes = true_classes
        self.num_classes = num_classes
        self.num_features = num_features
        node_types, edge_types = self.dataset.metadata()
        self.num_node_types = len(node_types)
        self.num_edge_types = len(edge_types)


    def summary(self):
        """
        Print a summary for current state.
        """
        print("GNNMutation - Summary ---------------------- ")
        node_types, edge_types = self.dataset.metadata()
        print("Number of node types:", self.num_node_types)
        for i in range(len(node_types)):
            print("Number of nodes for "+node_types[i]+ ":"+ str(len(self.dataset[node_types[i]].x)))

        print("Number of edge types:", self.num_edge_types)
        for i in range(len(edge_types)):
            print("Number of nodes for "+str(edge_types[i])+ ":"+ str((np.transpose(np.array(self.dataset[edge_types[i]].edge_index))).shape[0]))

        print("Number of classes:", self.num_classes)

        print("Early stopping:",str(self.early_stopping))
        print("Add SelfLoops:", str(self.add_selfloops))

        print("-------------------------------------------- ")


    # Train and test model in n-fold cross validation
    def evaluate_model_with_val(self, num_layers=NUM_GCN_LAYERS_BASE, num_gat_layers=NUM_GAT_LAYERS_BASE,
                       num_lin_layers=NUM_LIN_LAYERS_BASE, num_hidden_units=NUM_HIDDEN_UNITS,
                       num_gat_heads=NUM_GAT_HEADS_BASE, epoch_nr=NUM_EPOCHS_BASE, shuffle=True, weights=False,
                       learning_rate=LEARNING_RATE_BASE):
        print("Start TRAIN")
        dataset = self.dataset
        gene_names = self.gene_names
        patients = self.patients
        true_classes = self.true_classes
        num_classes = self.num_classes
        num_features = self.num_features
        early_stopping = self.early_stopping

        cross_val_test_acc_list = []
        cross_val_val_acc_list = []
        cross_val_test_f1score_list = []
        epoch_counts = []
        for i in range(NUM_REPEATS_FOR_CROSS_VAL):

            # prepare train-test masks
            train_masks, val_masks, test_masks = prepare_train_test_masks_for_cross_val_with_val(num_classes, true_classes)
            print("# Masks train-val-test ready #")
            #print(train_masks[0])
            #print(test_masks[0])

            test_acc_list = []
            val_acc_list = []
            test_f1score_list = []


            for j in range(CROSS_VAL_CONSTANT):

                if self.model_selection == MODEL_GCN:
                    model = modelGCN_Hetero(num_layers, num_lin_layers, num_features, num_hidden_units, num_classes,
                                            DROPOUT_RATE_BASE)
                elif self.model_selection == MODEL_GAT:
                    model = modelGAT_Hetero(num_gat_layers, num_lin_layers, num_features, num_hidden_units,
                                            num_gat_heads, num_classes, DROPOUT_RATE_BASE)

                print(model)
                start_train = datetime.datetime.now()

                # convert to heterogeneous model
                metadata = self.dataset.metadata()
                model = to_hetero(model, metadata, aggr="sum")
                # print(model)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = torch.nn.CrossEntropyLoss()
                print("# Model ready #")
                print("# Hetero #")


                # TRAIN
                epoch_nr = NUM_EPOCHS_BASE
                train_loss_list = []
                max_early_stopping_counter = MAX_EARLY_STOPPING_COUNTER
                last_min_loss = 1000000
                best_model = 0
                for epoch in range(epoch_nr):
                    model.train()
                    optimizer.zero_grad()

                    out = model(dataset.x_dict, dataset.edge_index_dict)
                    loss = criterion(out['patient'][train_masks[j]], dataset['patient'].y[train_masks[j]])
                    loss.backward()
                    optimizer.step()

                    # check for early-stopping
                    if early_stopping:
                        if loss <= last_min_loss:
                            last_min_loss = loss
                            early_stopping_counter = 0
                            best_model = copy.deepcopy(model)
                        else:
                            early_stopping_counter = early_stopping_counter + 1
                            if early_stopping_counter == max_early_stopping_counter:
                                model.load_state_dict(best_model.state_dict())
                                print("Early stooping at epoch ", epoch)
                                epoch_counts.append(epoch)
                                break

                    print('Epoch {}, train_loss {:.4f}'.format(epoch, loss))
                    train_loss_list.append(loss)

                # TEST
                model.eval()
                out = model(dataset.x_dict, dataset.edge_index_dict)
                # Use the class with highest probability.
                pred = out['patient'].argmax(dim=1)
                # Check against ground-truth labels. - val
                val_correct = torch.eq(pred[val_masks[j]], dataset['patient'].y[val_masks[j]])
                val_acc = int(torch.count_nonzero(val_correct)) / int(len(val_masks[j]))
                val_acc_list.append(val_acc)

                # Check against truth labels
                test_correct = torch.eq(pred[test_masks[j]], dataset['patient'].y[test_masks[j]])
                test_acc = int(torch.count_nonzero(test_correct)) / int(len(test_masks[j]))
                test_acc_list.append(test_acc)

                test_f1score = f1_score(dataset['patient'].y[test_masks[j]], pred[test_masks[j]])
                test_f1score_list.append(test_f1score)

                print('Repeat {}, fold {}, val_acc {:.4f} test_acc {:.4f}'.format(i, j, val_acc, test_acc))
                end_train = datetime.datetime.now()
                print("Fold time: ", end_train - start_train)


            avg_acc_for_repeat = sum(test_acc_list) / CROSS_VAL_CONSTANT
            cross_val_test_acc_list.append(avg_acc_for_repeat)

            avg_val_acc_for_repeat = sum(val_acc_list) / CROSS_VAL_CONSTANT
            cross_val_val_acc_list.append(avg_val_acc_for_repeat)

            avg_f1score_for_repeat = sum(test_f1score_list) / CROSS_VAL_CONSTANT
            cross_val_test_f1score_list.append(avg_f1score_for_repeat)

            print('Repeat {}, avg val_acc {:.4f} avg test_acc {:.4f} avg test_mcc {:.4f} avg test_auc {:.4f} avg test_auprc {:.4f}'.format(i, avg_val_acc_for_repeat, avg_acc_for_repeat, avg_f1score_for_repeat))


        avg_acc_overall = sum(cross_val_test_acc_list) / NUM_REPEATS_FOR_CROSS_VAL
        std_acc_overall = np.std(cross_val_test_acc_list)
        print('\nOVERALL avg test_acc {:.4f} ({:.4f})'.format(avg_acc_overall, std_acc_overall))

        avg_val_acc_overall = sum(cross_val_val_acc_list) / NUM_REPEATS_FOR_CROSS_VAL
        std_val_acc_overall = np.std(cross_val_val_acc_list)
        print('\nOVERALL avg val_acc {:.4f} ({:.4f})'.format(avg_val_acc_overall, std_val_acc_overall))

        avg_f1score_overall = sum(cross_val_test_f1score_list) / NUM_REPEATS_FOR_CROSS_VAL
        std_f1score_overall = np.std(cross_val_test_f1score_list)
        print('\nOVERALL avg test_f1score {:.4f} ({:.4f})'.format(avg_f1score_overall, std_f1score_overall))

        if len(epoch_counts) > 0:
            avg_epoch_count = sum(epoch_counts) / len(epoch_counts)
            print('\nOVERALL avg epoch_count {:.4f}'.format(avg_epoch_count))

        self.model_status = 'Trained'
        self.model = copy.deepcopy(model)
        torch.save(model.state_dict(), self.location + 'model/' + self.model_name)
