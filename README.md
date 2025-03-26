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

**Background:** We introduce a novel approach based on graph neural networks that jointly considers genetic mutations and protein interactions for cancer prediction. We use DNA mutations in whole exome sequencing data and create a heterogeneous graph in which patients and proteins are represented as nodes and protein-protein interactions as edges. Furthermore, the patient nodes are connected to the protein nodes based on the mutations in the patient’s DNA. With our new heterogeneous graph structure, the patient nodes are updated by both mutations and protein interactions. In addition, we calculated an importance vector for the mutations of each patient and used this as the feature value for the patient node. Since the effects of each mutation on disease development are different, we processed the input graph with attention-based graph neural networks. 

**Results:** We compiled a dataset from the UKBiobank consisting of patients with a cancer diagnosis as a case group and patients without a cancer diagnosis as a control group. We evaluated our approach for the four most common cancer types which are breast, prostate, lung and colon cancer and showed that the proposed framework effectively discriminates between case and control.

**Conclusions:** The results show that our proposed graph structure and node updating lead to better performance in cancer classification. We have also extended our system with an explainer. This reveals a list of causal genes for recognizing a cancer patient. We have found that some of them have already been investigated in cancer studies. This confirms that our system is able to recognize real causal genes for the selected cancer types and make predictions based on them.
