import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T


def load_dataset_hetero(feat_path="", similarity_path="", ppi_path= "", pg_path="", connected=True, threshold=950, undirected=False, add_selfloops = False, add_sim=False, file_prefix="general"):


    # Read in the features
    the_feats = pd.read_csv(feat_path, delimiter=",")
    patients = list(the_feats[the_feats.columns.values[0]])
    true_classes = the_feats[the_feats.columns.values[-1]]
    num_classes = len(np.unique(true_classes))
    genenames = list(the_feats.columns.values[1:-1])
    num_features = len(genenames)
    print("MY LEN PATIENT FEATS: "+str(len(genenames)))
    print("MY LEN PATIENTS: " + str(len(patients)))
    print("MY LEN GENES: " + str(len(genenames)))
    print("MY LEN CLASSES: "+str(num_classes))
    # nodes contains all nodes in the graph
    nodes = patients + genenames
    print("MY LEN ALL NODES: " + str(len(nodes)))

    # create feature list with shape [num_nodes, num_node_features]
    # for patients
    temp_feats = np.array(the_feats)
    temp_feats = temp_feats[:,1:-1]
    temp_feats = temp_feats.astype('f')

    # create dummy features for genes
    temp_feats_genes = np.identity(len(genenames))

    # create node dict for patients and genes
    new_nodes_patients = list(range(len(patients)))
    new_nodes_patients_dict = {old: new for old, new in zip(patients, new_nodes_patients)}
    new_nodes_genes = list(range(len(genenames)))
    new_nodes_genes_dict = {old: new for old, new in zip(genenames, new_nodes_genes)}


    # Read in the ppi network - gene 2 gene
    graph_path_ppi = ppi_path
    the_graph_ppi = pd.read_csv(graph_path_ppi, delimiter=" ")
    # prepare edges via ppi network (gene-gene)
    the_graph_ppi[the_graph_ppi.columns.values[0]] = the_graph_ppi[the_graph_ppi.columns.values[0]].map(new_nodes_genes_dict)
    the_graph_ppi[the_graph_ppi.columns.values[1]] = the_graph_ppi[the_graph_ppi.columns.values[1]].map(new_nodes_genes_dict)
    edge_index_ppi = the_graph_ppi[[the_graph_ppi.columns.values[0], the_graph_ppi.columns.values[1]]].to_numpy()
    # convert to a proper format and sort
    edge_index_ppi = np.array(sorted(edge_index_ppi, key=lambda x: (x[0], x[1]))).T
    print("MY LEN EDGES PPI: " + str(edge_index_ppi.shape[1]))

    # Read in the pg network - patient 2 gene , we reverse as gene  2 patient
    graph_path_pg = pg_path
    the_graph_pg = pd.read_csv(graph_path_pg, delimiter=" ")
    # prepare edges via ppi network (patient-gene)
    the_graph_pg[the_graph_pg.columns.values[0]] = the_graph_pg[the_graph_pg.columns.values[0]].map(new_nodes_patients_dict)
    the_graph_pg[the_graph_pg.columns.values[1]] = the_graph_pg[the_graph_pg.columns.values[1]].map(new_nodes_genes_dict)
    edge_index_pg = the_graph_pg[[the_graph_pg.columns.values[0], the_graph_pg.columns.values[1]]].to_numpy()
    edge_index_pg_rev = the_graph_pg[[the_graph_pg.columns.values[1], the_graph_pg.columns.values[0]]].to_numpy()
    # convert to a proper format and sort
    edge_index_pg = np.array(sorted(edge_index_pg, key=lambda x: (x[0], x[1]))).T
    edge_index_pg_rev = np.array(sorted(edge_index_pg_rev, key=lambda x: (x[0], x[1]))).T
    print("MY LEN EDGES PG: " + str(edge_index_pg.shape[1]))


    dataset = HeteroData()
    dataset['patient'].x = torch.tensor(temp_feats, dtype=torch.float32)        # [num_patients, num_features_patient]
    dataset['gene'].x = torch.tensor(temp_feats_genes, dtype=torch.float32)     # [num_genes, num_features_gene]
    dataset['gene', 'mutate', 'patient'].edge_index = torch.tensor(edge_index_pg_rev, dtype=torch.int64)            # [2, num_edges_mutates] pg
    dataset['gene', 'interact', 'gene'].edge_index = torch.tensor(edge_index_ppi, dtype=torch.int64)                # [2, num_edges_interacts] ppi
    dataset.num_classes = torch.tensor(num_classes, dtype=torch.int64)
    dataset['patient'].y = torch.tensor(true_classes, dtype=torch.int64)


    #print(dataset)
    dataset = T.ToUndirected()(dataset)
    if not undirected:
        del dataset['patient', 'rev_mutate', 'gene']

    if add_selfloops:
        # for patients
        patient_selfloops_list = [[i,i] for i in range(len(patients))]
        edge_index_patient_selfloops = np.array(sorted(patient_selfloops_list, key=lambda x: (x[0], x[1]))).T
        dataset['patient', 'update1', 'patient'].edge_index = torch.tensor(edge_index_patient_selfloops, dtype=torch.int64)  # [2, num_patients]

        # for genes
        gene_selfloops_list = [[i, i] for i in range(len(genenames))]
        edge_index_gene_selfloops = np.array(sorted(gene_selfloops_list, key=lambda x: (x[0], x[1]))).T
        dataset['gene', 'update2', 'gene'].edge_index = torch.tensor(edge_index_gene_selfloops, dtype=torch.int64)  # [2, num_genes]

    return dataset, genenames, patients, true_classes, num_features, num_classes
