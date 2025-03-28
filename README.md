# GNNMutation: a heterogeneous graph-based framework for cancer detection

This repository provides the source-code and a set of synthetic inputs for the GNNMutation method proposed in the study.
You can try the code with the command below:

python run_GNNMutationHetero.py

System requirements:
python,
pytorch,
scikit-learn

**Please cite:**
Özcan Şimşek, N.Ö., Özgür, A., & Gürgen, F.S. (2025). GNNMutation: a heterogeneous graph-based framework for cancer detection. BMC Bioinformatics.

**Abstract**

**Background:** When genes are translated into proteins, mutations in the gene sequence can lead to changes in protein structure and function as well as in the interactions between proteins. These changes can disrupt cell function and contribute to the development of tumors. In this study, we introduce a novel approach based on graph neural networks that jointly considers genetic mutations and protein interactions for cancer prediction. We use DNA mutations in whole exome sequencing data and construct a heterogeneous graph in which patients and proteins are represented as nodes and protein-protein interactions as edges. Furthermore, patient nodes are connected to protein nodes based on mutations in the patient’s DNA. Each patient node is represented by a feature vector derived from the mutations in specific genes. The feature values are calculated using a weighting scheme inspired by information retrieval, where whole genomes are treated as documents and mutations as words within these documents. The weighting of each gene, determined by its mutations, reflects its contribution to disease development. The patient nodes are updated by both mutations and protein interactions within our noval heterogeneous graph structure. Since the effects of each mutation on disease development are different, we processed the input graph with attention-based graph neural networks.

**Results:** We compiled a dataset from the UKBiobank consisting of patients with a cancer diagnosis as the case group and those without a cancer diagnosis as the control group. We evaluated our approach for the four most common cancer types, which are breast, prostate, lung and colon cancer, and showed that the proposed framework effectively discriminates between case and control groups.

**Conclusions:** The results indicate that our proposed graph structure and node updating strategy improve cancer classification performance. Additionally, we extended our system with an explainer that identifies a list of causal genes which are effective in the model’s cancer diagnosis predictions. Notably, some of these genes have already been studied in cancer research, demonstrating the system’s ability to recognize causal genes for the selected cancer types and make predictions based on them.
