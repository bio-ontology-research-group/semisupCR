# semisupCR
semi-supervised concept recognition

STEPS:
1. create an conda environment 
2. Install biobert-embeddings from https://pypi.org/project/biobert-embedding/
3. Update embedding.py (provided from this repository). It is modified to handle the word vector generation from biobert tokens as input (the original version generates word vectors from text). This is needed for the truncation of the tokens as originnal bert can handle max 512 tokens
4. run python biobert_clst_train.py >out.txt to print the word vectors for positive and negative classes.
