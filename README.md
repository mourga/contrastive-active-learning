# Contrastive Active Learning (CAL)
> [**Active Learning by Acquiring Contrastive Examples**](TBA)  
> Katerina Margatina, Giorgos Vernikos, Loic Barrault, Nikolaos Aletras  
> Empirical Methods in Natural Language Processing (EMNLP) 2021.


---
# Quick Overview

This repository contains code for running active learning with our proposed acquisition function, CAL, and other baselines. 

### Acquisition functions 
Specifically, there is code for running active learning with the following acquisition functions: `CAL`, `Entropy`, `Least Confidence`, `BALD`, `BatchBALD`, `ALPS`, `BADGE`, `BertKM` and `Random sampling`.
### Tasks & Datasets 
We evaluate the aforementioned AL algorithms in 4 Natural Lagnuage Processing (NLP) tasks and 7 datasets.
- Sentiment analysis: `SST-2`, `IMDB`
- Topic classification: `AGNEWS`, `DBPEDIA`, `PUBMED`
- Natural language inference: `QNLI`
- Paraphrase detection: `QQP`
### Models
So far we have used only `BERT-BASE`, but the code can support any other model (e.g. from [HuggingFace](https://github.com/huggingface/transformers)) with minimal changes.

---
## Installation
This project is implemented with Python 3 and PyTorch 1.2.0.

**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n cal python=3.7
conda activate cal
```

Also install the required torch package:

```
conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
```
Finally install the rest of the requirements:

```
pip install -r requirements.txt
```
Please check [here](https://pytorch.org/get-started/previous-versions/) for information on how to install properly the required pytorch version for your machine.

---
## Download data
In order to download the datasets we used run the following script:
```
bash get_data.sh
```
DBPedia is too large so dowload it manually from [here](https://drive.google.com/uc?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&export=download).

---
## Organization

---
## Usage

---

## Acknowledgements

We would like to thank the community for releasing their code! This repository contains code from [HuggingFace](https://github.com/huggingface/transformers), from the [ALPS](https://github.com/forest-snow/alps) repository, from the [BatchBALD](https://github.com/BlackHC/BatchBALD) repository.

---
## Contact
Please feel free to contact me in case you require any help setting up the repo!:blush:
