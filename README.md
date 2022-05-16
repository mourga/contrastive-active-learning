# Contrastive Active Learning (CAL) 

<!-- ## ⚠️WORK IN PROGRESS⚠️ -->

> [**Active Learning by Acquiring Contrastive Examples**](https://aclanthology.org/2021.emnlp-main.51/)  
> Katerina Margatina, Giorgos Vernikos, Loic Barrault, Nikolaos Aletras  
> Empirical Methods in Natural Language Processing (EMNLP) 2021.


---
# Quick Overview

In our paper, we propose a new _acquisition function for active learning_ namely **CAL: Contrastive Active Learning**. This repository contains code for running active learning with our proposed acquisition function, CAL, and other baselines. 

<p align="center">
  <img src="cal.png" width="500">
</p>

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
This project is implemented with `Python 3`, `PyTorch 1.9.0` and `transformers 3.1.0`.

**Create Environment (Optional):**  Ideally, you should create a conda environment for the project.

```
conda create -n cal python=3.7
conda activate cal
```

Also install the required torch package(*):

```
conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
<!--  conda install pytorch==1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia ford -->
Finally install the rest of the requirements:

```
pip install -r requirements.txt
```
(*) Please check [here](https://pytorch.org/) for information on how to install properly the required pytorch version for your machine (cuda). This is important! Do not copy paste the above line without first checking which cuda version is supported by your machine. You can run `nvcc --version` in your terminal to check it.

---
## Download data
In order to download the datasets we used run the following script:
```
bash get_data.sh
```
`DBPedia` is too large so dowload it manually from [here](https://drive.google.com/uc?id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k&export=download).

---
## Organization
The repository is organizes as follows:
- `acquisition`: implementation of acquisition functions
- `analysis`: scrips for analysis (see §6 of our [paper](https://arxiv.org/pdf/2109.03764.pdf))
- `cache`: models downloaded from `HuggingFace`
- `checkpoints`: model checkpoints
- `data`: datasets
- `utilities`: scripts for helpers (e.g. data loaders and processors)

---
## Usage

The main script to run any AL experiment is `run_al.py`. 

Example:
```
python run_al.py --dataset_name sst-2 --acquisition cal
```
---

## Acknowledgements

We would like to thank the community for releasing their code! This repository contains code from [HuggingFace](https://github.com/huggingface/transformers),  [ALPS](https://github.com/forest-snow/alps), and [BatchBALD](https://github.com/BlackHC/BatchBALD) repositories.


---
## Reference
Please feel free to cite our paper if you use our code or proposed algorithm.:blush:
```
@inproceedings{margatina-etal-2021-active,
    title = "Active Learning by Acquiring Contrastive Examples",
    author = {Margatina, Katerina  and
      Vernikos, Giorgos  and
      Barrault, Lo{\"\i}c  and
      Aletras, Nikolaos},
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.51",
    pages = "650--663",
    abstract = "Common acquisition functions for active learning use either uncertainty or diversity sampling, aiming to select difficult and diverse data points from the pool of unlabeled data, respectively. In this work, leveraging the best of both worlds, we propose an acquisition function that opts for selecting contrastive examples, i.e. data points that are similar in the model feature space and yet the model outputs maximally different predictive likelihoods. We compare our approach, CAL (Contrastive Active Learning), with a diverse set of acquisition functions in four natural language understanding tasks and seven datasets. Our experiments show that CAL performs consistently better or equal than the best performing baseline across all tasks, on both in-domain and out-of-domain data. We also conduct an extensive ablation study of our method and we further analyze all actively acquired datasets showing that CAL achieves a better trade-off between uncertainty and diversity compared to other strategies.",
}
```

---
## Contact
Please feel free to raise an issue or contact me in case you require any help setting up the repo!:blush:
