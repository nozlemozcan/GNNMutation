# GNNMutation: a heterogeneous graph-based framework for cancer detection

**Please cite:**
Özcan Şimşek, N.Ö., Özgür, A., & Gürgen, F.S. (2025). GNNMutation: a heterogeneous graph-based framework for cancer detection. BMC Bioinformatics.


# Details of the Input Data

There are 3 input files for the GNNMutation model.\
Each file includes a header line which is followed by multiple data lines.

## 1. the PPI network file
> ppi_network_pyhsical_String_breast.txt

This file includes the PPI network structure for the heterogeneous graph of GNNMutation.\
The header is:
> protein1 protein2 combined_score

This is a space delimited file.\
The source for this input file is the physical PPI network in StringDB ( https://string-db.org/ ).\
We selected the subset of the whole PPI based on our list of mutated proteins.


## 2. the patient mutation features file
> xfullgene_bm25_tfrf_breast_small200_neweid.csv

This file includes the patient mutation features for the patient nodes in the heterogeneous graph of GNNMutation.\
The header is:
> eid, protein1, protein2, ... proteinN, Class

This is a comma delimited file.\
The feature values for proteins are calculated with BM25-tf-rf metric.\
   
## 3. the protein-patient network file
> pg_network_breast_small200_max300_neweid.csv

This file includes the protein-patient network structure for the heterogeneous graph of GNNMutation.\
The header is:
> node1 node2 edge_weight

This is a space delimited file.\
The content of this file is selected based on the patient mutation features file.\

