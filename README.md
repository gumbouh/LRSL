# LRSL
This repository provides the implementation of **Logits Reranking via Semantic Labels for Hard Samples in Text
Classification (LRSL)**, a method designed to enhance text classification performance, especially for hard samples, by leveraging label semantics and similarity-based reranking.

# Key Features
1.	Model-agnostic: Works with any pre-trained language model (e.g., BERT, RoBERTa, Llama).
2.	Hard Sample Detection: Automatically identifies challenging samples with a variance-based metric.
3.	Semantic Reranking: Refines logits of hard samples using semantic distances computed from fine-tuned embeddings.
# Steps to Reproduce
## Step 1: Prepare a Fine-Tuned Model

Use a well-trained classification model, fine-tuned on your dataset.

## Step 2: Extract Top-k Logits
Perform inference on the training dataset to extract the top-k logits for each sample. These logits serve as a basis for fine-tuning the text embeddings.

## Step 3: Fine-Tune Text Embedding
Utilize the training dataset to fine-tune a text embedding model. The embedding is trained to map text samples and semantic labels into a shared representation space using a triplet-based loss.
## Step 4: Perform Similarity-Based Reranking
### 4.1 Compute Variance for Hard Sample Detection
Use a portion of the dataset (e.g., training or validation set) to calculate the variance (Var) of logits distributions as a measure of confidence.
### 4.2 Rerank Test Samples
- Automatically detect hard samples in the test dataset based on the sigma^2_mis metric.
- Rerank the logits of these samples using their cosine similarity with fine-tuned text embeddings for improved classification accuracy.
## File Description
### datasets

### finetune-text-embedding

### llms-cls-qlora

### bert-cls

# Citation
```
@inproceedings{huang-etal-2024-logits,
    title = "Logits Reranking via Semantic Labels for Hard Samples in Text Classification",
    author = "Huang, Peijie  and
      Huang, Junbao  and
      Xu, Yuhong  and
      Li, Weizhen  and
      Xiao, Xisheng",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.657",
    doi = "10.18653/v1/2024.findings-emnlp.657",
    pages = "11250--11262",
}
```