# Virtial Cell Challange
The source for the challange is [here](https://virtualcellchallenge.org/)

A [blog describing](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge) the wetlab background, dataset genaration and scoring. There was some mindor problem in the original scoring description, now (~Oct. 17. ) it is corrected. 

## The goal
### Note: Shortened version of Hugging Face description. 
Train a model capable of predicting the effect of silencing a gene in a (partially) unseen cell type, a task they term context generalization.  

The training set consists of a sparse matrix and some associated metadata. More specifically, we have 220k cells, and for each cell we have a transcriptome. This transcriptome is a sparse row vector, where each entry is the raw count of RNA molecules (transcripts) that the corresponding gene (our column) encodes for. Of the 220k cells, ~38k are unperturbed, meaning no gene has been silenced using CRISPR. The inability to measure the cell state before and after introduces many issues, as we are forced to use a population of basal (a.k.a control, unperturbed) cells as a reference point. The control cells and perturbed cells are not entirely homogeneous even prior to the perturbation. This means that we have to now separate out our true signal, the perturbation, from noise induced by the heterogeneity.  

We have input as cells described by their genes from the same population, and the perturbations encoded in OHE. The target is predict the perturbed transcriptome.

## The model
### State Transition Model (ST)
The State Transition Model is a relatively simple transformer with a Llama backbone that operates upon the following:

1. A set of transcriptomes (or SE embeddings) for covariate matched basal cells.
2. A set of one hot vectors representing our gene perturbation for each cell.

Predicts the the perturbed transcriptome. ST is trained using Maximum Mean Discrepancy. Put simply, the model learns to minimize the difference between the two probability distributions.

### State Embeddig Model (SE)
A cell embedding is created form amino acid sequences with a specific embedding model ESM2. This creates an embedding per amino acid and they are mean pooled to obtain a protein isomorf embedding. These protein isomorfs can be mean pooled together and we get a gene embedding. This is projected the gene embeddings into the model dimensions using an encode block. A cell is represented by as the top 2048 genes ranked by log fold expression level.
