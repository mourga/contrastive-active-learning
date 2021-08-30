import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

glue_datasets = ['sst-2', 'mrpc', 'qnli', "cola", "mnli", "mnli-mm", "sts-b", "qqp", "rte", "wnli"]
available_datasets = glue_datasets + ["ag_news", "trec-6", "dbpedia", "imdb", "pubmed"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

DATA_DIR = os.path.join(BASE_DIR, 'data')

GLUE_DIR = os.path.join(DATA_DIR, 'glue_data')

EXP_DIR = os.path.join(BASE_DIR, 'experiments')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')

ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')

AWD_DIR = os.path.join(CKPT_DIR, 'lms')

acquisition_functions = ["random", "bald", "batch_bald", "entropy", "least_conf", "bkm", "badge", "constrastive"]

# acquistion2str = {'bald': 'BALD', 'batch_bald': 'BatchBALD', 'entropy': 'Max Entropy',
#                   'least_conf': 'Least Confidence', 'random': 'Random'}
#
# train_size_dict = {'sst-2': 67349,
#                    'mrpc': 3668,
#                    'qnli': 104743,
#                    'ag_news': 120000,
#                    'trec-6': 5452,
#                    'dbpedia': 560000,
#                    'imdb': 128000,
#                    'rte': 2000,
#                    'qqp': 300000,
#                    'mnli': 1000}

