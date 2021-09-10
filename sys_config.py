import os

import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("Cuda available:", torch.cuda.is_available())

# print("torch:", torch.__version__)
# print("Cuda:", torch.backends.cudnn.cuda)
# print("CuDNN:", torch.backends.cudnn.version())

glue_datasets = ['sst-2', 'qnli', "qqp"]
available_datasets = glue_datasets + ["ag_news", "dbpedia", "imdb", "pubmed"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = os.path.join(BASE_DIR, 'data')

EXP_DIR = os.path.join(BASE_DIR, 'experiments')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')

ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

acquisition_functions = ["random", "bald", "batch_bald", "entropy", "least_conf", "FTbertKM", "badge", "cal"]


