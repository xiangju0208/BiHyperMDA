# BiHyperMDA: Bidirectional Hypergraph Learning with Hyperedge Interactions for miRNA-Disease Association Prediction 

Identification of miRNA-disease associations (MDA) is a critical task for understanding disease mechanisms and developing diagnostic biomarkers, while traditional experimental approaches are notoriously costly and time-consuming. Many computational algorithms have emerged as an efficient and scalable alternative for reliably inferring potential MDAs, but the sparsity of known miRNA–disease associations make models relying solely on supervised signals prone to overfitting and limits their generalization ability particularly in predicting associations involving previously unseen miRNAs and diseases, and existing methods typically extract high-order information through the node–hyperedge–node paradigm while ignoring interactions among hyperedges and treating all hyperedges as equally important and independent. To address these issues, we propose a novel end-to-end model called BiHyperMDA, which constructs similarity-based and co-occurrence-based hypergraphs and performs bidirectional hypergraph convolution from both node and hyperedge perspectives to capture high-order information. Furthermore, synergistic regulatory patterns are extracted by combining multiple hyperedges. Finally, the prediction scores are integrated with a bidirectional graph diffusion prior to improve the robustness and accuracy of prediction, while alleviating the cold-start problem. Experimental results demonstrate the effectiveness of distinct modules, and the good performance of our model compared to state-of-the-art models. This work provides a useful approach for computationally identifying potential MDAs. 
 
<div align="center">
  <img src="Fig.1-BiHyperMDA.png" width="80%" />
</div>

## Requirements
  * Python 3.9 or higher
  * PyTorch 2.0.0 
  * torch-geometric 2.3.0
  * numpy 1.26.4
  * scikit-learn 1.1.3
## Usage
### 1. Data Preprocessing 
For detailed implementation instructions, please refer to getData.py 
### 2. Get miRNA and disease embedding 
For detailed implementation instructions, please refer to model.py
### 3. Predicting and Model Output 
We use AUROC and AUPRC as two standard evaluation metrics. 
*For detailed implementation instructions, please refer to model.py and train.py
## Quick Run
Execute python main.py to run the code

## Cite
If you use the work in your research, please cite:
* Xiang, et al., BiHyperMDA: Bidirectional Hypergraph Learning with Hyperedge Interactions for miRNA-Disease Association Prediction, BIBM 2026.

## Contact
* Please contact us for any further questions:
* Email: Ju Xiang xiang.ju@foxmail.com
