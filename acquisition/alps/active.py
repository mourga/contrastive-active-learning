"""
Code from https://github.com/forest-snow/alps
"""

import logging
import os
import sys

import numpy as np
import torch
from torch.nn.functional import normalize

# from src.data import processors
# from src import setup, train, sample, cluster
sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from acquisition.alps.cluster import kmeans, kmeans_pp, badge, kcenter
from acquisition.alps.sample import get_scores_or_vectors

logger = logging.getLogger(__name__)


def cluster_method(sampling):
    """Given the [sampling] method for active learning, return clustering function [f]
     and [condition], boolean that indicates whether sampling
    is conditioned on prior iterations"""
    if "KM" in sampling:
        f = kmeans
        condition = False
    elif "KP" in sampling:
        f = kmeans_pp
        condition = True
    elif "FF" in sampling:
        f = kcenter
        condition = True
    elif sampling == "badge":
        f = badge
        condition = False
    elif sampling == "alps":
        f = kmeans
        condition = False
    else:
        #  [sampling] is not cluster-based strategy
        f = None
        condition = None
    return f, condition

def acquire(pool, sampled, args, model, tokenizer, original_inds=None):
    """Sample data from unlabeled data [pool].
    The sampling method may need [args], [model], [tokenizer], or previously
    [sampled] data."""
    scores_or_vectors = get_scores_or_vectors(pool, args, model, tokenizer)
    clustering, condition = cluster_method(args.sampling)
    if original_inds is not None:
        _sampled = [i for i,v in enumerate(original_inds) if v in sampled]  # indexing from initial -> 20K
        if _sampled == []:
            _sampled = torch.LongTensor([])
        else:
            _sampled = torch.LongTensor(_sampled)
        assert len(_sampled) == len(sampled)
    else:
        _sampled = sampled
    unsampled = np.delete(torch.arange(len(pool)), _sampled)
    # if original_inds is not None:
    #     not_available_inds = [x for x in np.arange(len(pool)) if x not in original_inds]
    #     cannot_sample_inds = not_available_inds + sampled
    if clustering is not None:
        # cluster-based sampling method like BADGE and ALPS
        vectors = normalize(scores_or_vectors)
        centers = _sampled.tolist()
        if not condition:
            # do not condition on previously chosen points
            queries_unsampled = clustering(
                vectors[unsampled], k = args.query_size
            )
            # add new samples to previously sampled list
            queries = centers + (unsampled[queries_unsampled]).tolist()
        else:
            queries = clustering(
                vectors,
                k = args.query_size,
                centers = centers
            )
        queries = torch.LongTensor(queries)
    else:
        # scoring-based methods like maximum entropy
        scores = scores_or_vectors
        _, queries_unsampled = torch.topk(scores[unsampled], args.query_size)
        queries = torch.cat((_sampled, unsampled[queries_unsampled]))
    assert len(queries) == len(queries.unique()), "Duplicates found in sampling"
    assert len(queries) > 0, "Sampling method sampled no queries."
    # return queries
    # assert len([x for x in queries.cpu().numpy() if x in original_inds])==len(original_inds), 'original {}, sampled {}'.format(original_inds, queries.cpu().numpy())
    return queries.cpu().numpy()

# def main():
#     args = setup.get_args()
#     setup.set_seed(args)
#
#     print(args)
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
#     )

#     args.task_name = args.task_name.lower()
#     if args.task_name not in processors:
#         raise ValueError("Task not found: %s" % (args.task_name))
#     # first, get already sampled points
#     sampled_file = os.path.join(args.output_dir, 'sampled.pt')
#     if os.path.isfile(sampled_file):
#         sampled = torch.load(sampled_file)
#     else:
#         sampled = torch.LongTensor([])
#
#     # decide which model to load based on sampling method
#     args.head = sampling_to_head(args.sampling)
#     if args.head == "lm":
#         # load pre-trained model
#         args.model_name_or_path = args.base_model
#     model, tokenizer, _, _= setup.load_model(args)
#
#
#     dataset = train.load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
#
#     logger.info(f"Already sampled {len(sampled)} examples")
#     sampled = acquire(dataset, sampled, args, model, tokenizer)
#
#
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     torch.save(sampled, os.path.join(args.output_dir, 'sampled.pt'))
#     logger.info(f"Sampled {len(sampled)} examples")
#     return len(sampled)
#
# if __name__ == "__main__":
#     main()
