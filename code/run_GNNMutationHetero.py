import sys
import datetime
from GNNMutationHetero import GNNMutationHetero
from gnn_constants_hetero import *
import numpy as np

start = datetime.datetime.now()
print("Current date and time start: "+start.strftime("%Y-%m-%d %H:%M:%S") + "\n")

model_name="GNNMutation_model.pth"
similarity_file=""

#read input
main_folder = "./data/"    # input folder
features_file = main_folder+"xfullgene_bm25_tfrf_breast_small200.csv"        # input features file
ppi_file = main_folder+"ppi_network_pyhsical_String_breast.txt"              # input ppi edges file
pg_file = main_folder+"pg_network_breast_small200_max300.csv"                # input patient-protein edges file
print(features_file)
model_selection = "gat"

file_prefix="all"
cutoff = EDGE_WEIGHT_CUTOFF_BASE
undirected = UNDIRECTED_FLAG
early_stopping = EARLY_STOPPING_FLAG
G = GNNMutationHetero(model_selection, file_prefix,main_folder,features_file,similarity_file,ppi_file, pg_file, cutoff, undirected, early_stopping, model_name)
G.summary()
G.evaluate_model_with_val()

end = datetime.datetime.now()
print("\n")
print("\n")
print("Current date and time start: "+start.strftime("%Y-%m-%d %H:%M:%S"))
print("Current date and time end: "+end.strftime("%Y-%m-%d %H:%M:%S") + "\n")
print(features_file)
