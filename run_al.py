import collections
import json
import math
import os
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric
from torch import nn
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import set_seed

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sys_config import acquisition_functions, CACHE_DIR, glue_datasets, GLUE_DIR, DATA_DIR, CKPT_DIR
from utilities.general import create_dir

logger = logging.getLogger(__name__)

def al_loop(args):
    """
    Main script for the active learning algorithm.
    :param args: contains necessary arguments for model, training, data and AL settings
    :return:
    Datasets (lists): X_train_original, y_train_original, X_val, y_val
    Indices (lists): X_train_init_inds - inds of first training set (iteration 1)
                     X_train_current_inds - inds of labeled dataset (iteration i)
                     X_train_remaining_inds - inds of unlabeled dataset (iteration i)
                     X_train_original_inds - inds of (full) original training set
    """
    #############
    # Setup
    #############
    # Set the random seed manually for reproducibility.
    set_seed(args.seed)

    ##############################################################
    # Load data
    ##############################################################
    X_test_ood = None
    X_train_original, y_train_original = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                          evaluate=False)
    if args.task_name == 'imdb' and os.path.exists(IMDB_CONTR_DATA_DIR):
        X_val_contrast, y_val_contrast = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                          evaluate=True, contrast=True)
        X_test_contrast, y_test_contrast = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                            test=True, contrast=True)
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)
    if args.task_name == 'mnli':
        X_test_ood, y_test_ood = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True,
                                                  ood=True)
    if args.task_name == 'sentiment':
        X_test_ood_amazon, y_test_ood_amazon = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                                test=True, ood=True, ood_name="amazon")
        X_test_ood_yelp, y_test_ood_yelp = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                            test=True, ood=True, ood_name="yelp")
        X_test_ood_semeval, y_test_ood_semeval = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type,
                                                                  test=True, ood=True, ood_name="semeval")
        X_test_ood = X_test_ood_amazon
    if args.task_name == 'imdb':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(GLUE_DIR, 'SST-2'), 'sst-2', args.model_type,
                                                  test=True)
    if args.task_name == 'sst-2':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'IMDB'), 'imdb', args.model_type,
                                                  test=True)
    if args.task_name == 'qqp':
        # X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(GLUE_DIR, 'MRPC'), 'mrpc', args.model_type, test=True)
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(DATA_DIR, 'TwitterPPDB'), 'twitterppdb',
                                                  args.model_type, test=True)
    if args.task_name == 'mrpc':
        X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(GLUE_DIR, 'QQP'), 'qqp', args.model_type,
                                                  test=True)
    # if args.task_name == 'qnli':
    #     X_test_ood, y_test_ood = get_glue_dataset(args, os.path.join(GLUE_DIR, 'WNLI'), 'wnli', args.model_type, test=True)

    X_train_original_inds = list(np.arange(len(X_train_original)))  # original pool
    X_val_inds = list(np.arange(len(X_val)))
    X_test_inds = list(np.arange(len(X_test)))

    # if args.dataset_name in ['dbpedia', 'qqp']:
    if args.dataset_name in ['dbpedia']:
        # undersample dpool up to 20K + dval up to 2K
        new_X_train_original_inds, X_train_discarded_inds, _, _ = train_test_split(X_train_original_inds,
                                                                                   y_train_original,
                                                                                   train_size=20000,
                                                                                   random_state=42,
                                                                                   stratify=y_train_original)

        new_X_val_inds, X_val_discarded_inds, _, _ = train_test_split(X_val_inds,
                                                                      y_val,
                                                                      train_size=2000,
                                                                      random_state=42,
                                                                      stratify=y_val)
        X_train_original_inds = new_X_train_original_inds
        X_val_inds = new_X_val_inds

    if args.debug:
        size = 500
        X_train_original_inds = X_train_original_inds[:size]
        X_val_inds = X_val_inds[:size]
        X_val = X_val[:size]
        y_val = y_val[:size]
        X_test = X_test[:size]
        X_test_inds = X_test_inds[:size]
        y_test = y_test[:size]
        # X_train_original_inds = X_train_original_inds[:100]
        # debug_inds_tr = np.random.randint(low=0, high=len(X_train_original), size=(50))
        # X_train_original_inds = debug_inds_tr
        # debug_inds_val = np.random.randint(low=0, high=len(X_val), size=(50))
        # debug_inds_test = np.random.randint(low=0, high=len(X_test), size=(50))
        # X_train_original=np.array(X_train_original)[debug_inds_tr]
        # y_train_original=np.array(y_train_original)[debug_inds_tr]
        # X_val=np.array(X_val)[debug_inds_val]
        # y_val=np.array(y_val)[debug_inds_val]
        # X_test=np.array(X_test)[debug_inds_test]
        # y_test=np.array(y_test)[debug_inds_test]

    args.binary = True if len(set(np.array(y_train_original)[X_train_original_inds])) == 2 else False
    args.num_classes = len(set(np.array(y_train_original)[X_train_original_inds]))

    if args.acquisition_size is None:
        args.acquisition_size = round(len(X_train_original_inds) / 100)  # 1%
        if args.dataset_name in ['qnli', 'ag_news']:
            args.acquisition_size = round(args.acquisition_size / 2)  # 0.5%
        # elif args.dataset_name in ['dbpedia']:
        #     args.acquisition_size = round(len(X_train_original_inds) / 1000)  # 0.1%
    if args.init_train_data is None:
        args.init_train_data = round(len(X_train_original_inds) / 100)  # 1%
        if args.dataset_name in ['qnli', 'ag_news']:
            args.init_train_data = round(args.init_train_data / 2)  # 0.5%
        # elif args.dataset_name in ['dbpedia']:
        #     args.init_train_data = round(len(X_train_original_inds) / 1000)  # 0.1%

    if args.indicator == "small_config":
        args.acquisition_size = 100
        args.init_train_data = 100
        args.budget = 5100

    if args.indicator == "25_config":
        args.acquisition_size = round(len(X_train_original_inds) * 2 / 100)  # 2%
        args.init_train_data = round(len(X_train_original_inds) * 1 / 100)  # 1%
        args.budget = round(len(X_train_original_inds) * 17 / 100)  # 25%
        if args.counterfactual is not None:
            if args.dataset_name == 'sentiment':
                args.acquisition_size = 100
                args.init_train_data = 100
                args.budget = 1100

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    assert bool(not set(X_train_current_inds) & set(X_train_remaining_inds))  # assert no same data in Dtrain and Dpool

    # For the contrastive acquisition we also need the model's represenations, apart from the logits.
    return_repr = True if acq_fun in ['contrastive'] else False

    # Dicts to store results
    it2per = {}  # iterations to data percentage
    results_per_iteration = {}
    current_iteration = 1
    current_annotations = len(X_train_current_inds)

    #############
    # Start AL!
    #############
    while current_iteration < total_iterations + 1:

        it2per[str(current_iteration)] = round(len(X_train_current_inds) / len(X_train_original_inds), 2) * 100

        ##############################################################
        # Train model on training dataset (Dtrain)
        ##############################################################
        print("\n Start Training model of iteration {}!\n".format(current_iteration))
        train_results = train(X_train_current_inds,
                              X_train_original,
                              y_train_original,
                              X_val_inds,
                              X_val,
                              y_val,
                              iteration=current_iteration,
                              )

        model = train_results['model']  # the saved ckpt - best model from training
        val_acc_previous = train_results['val_acc']  # accuracy in the validation set of best trained model
        print("\nDone Training!\n")

        ##############################################################
        # Test model on test data (D_test) - optional
        ##############################################################
        test_results, test_logits = None, None
        if X_test is not None:
            print("\nStart Testing on test set!\n")
            test_results, test_logits = evaluate(X_inds=X_test_inds,
                                                 X=X_test,
                                                 y=y_test,
                                                 model=model,
                                                 return_repr=False)

        ##############################################################
        # Test model on unlabeled data (Dpool)
        ##############################################################
        print("\nEvaluating Dpool!\n")
        dpool_loss, logits_dpool, results_dpool = [], [], []
        if acq_fun not in ['random', 'alps', 'badge', 'FTbertKM']:
            results_dpool, logits_dpool = evaluate(X_inds=X_train_remaining_inds,
                                                   X=X_train_original,
                                                   y=y_train_original,
                                                   model=model,
                                                   return_repr=return_repr)

        ##############################################################
        # Select unlabeled samples for annotation
        # -> annotate
        # -> update training dataset & unlabeled dataset
        ##############################################################
        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        if acq_fun == "contrastive":
            # Forward pass on current train set to get logits and representations
            results_dtrain, logits_dtrain = evaluate(X_inds=X_train_current_inds,
                                                     X=X_train_original,
                                                     y=y_train_original,
                                                     model=model,
                                                     return_repr=return_repr)

            sampled_ind, stats = contrastive_acq_fun(X_original=X_train_original,
                                                     y_original=y_train_original,
                                                     original_inds=X_train_original_inds,
                                                     labeled_inds=X_train_current_inds,
                                                     candidate_inds=X_train_remaining_inds,
                                                     logits_dpool=logits_dpool, train_logits=logits_dtrain,
                                                     dtrain_reprs=results_dtrain["representations"],
                                                     dpool_reprs=results_dpool["representations"],
                                                     acq_size=acq_size, num_nei=10)
        else:
            sampled_ind, stats = calculate_uncertainty(acq_fun=acq_fun,
                                                       logits=logits_dpool,
                                                       acq_size=acq_size,
                                                       candidate_inds=X_train_remaining_inds,
                                                       labeled_inds=X_train_current_inds,
                                                       original_inds=X_train_original_inds,
                                                       X_original=X_train_original, y_original=y_train_original)

        # Update results dict
        results_per_iteration[str(current_iteration)] = {'data_percent': it2per[str(current_iteration)],
                                                         'total_train_samples': len(X_train_current_inds),
                                                         }
        results_per_iteration[str(current_iteration)]['val_results'] = train_results
        results_per_iteration[str(current_iteration)]['test_results'] = test_results

        results_per_iteration[str(current_iteration)].update(stats)

        current_annotations += acq_size

        # X_train_current_inds and X_train_remaining_inds are lists of indices of the original dataset
        # sampled_inds is a list of indices OF THE X_train_remaining_inds(!!!!) LIST THAT SHOULD BE REMOVED
        # INCEPTION %&#!@***CAUTION***%&#!@
        if acq_fun in ['alps', 'badge', 'adv', 'FTbertKM', 'contrastive']:
            X_train_current_inds += list(sampled_ind)
        else:
            X_train_current_inds += list(np.array(X_train_remaining_inds)[sampled_ind])

        if acq_fun in ['alps', 'badge', 'adv', 'FTbertKM', 'contrastive']:
            selected_dataset_ids = sampled_ind
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json
        else:
            selected_dataset_ids = list(np.array(X_train_remaining_inds)[sampled_ind])
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json

        if acq_fun in ['alps', 'badge', 'adv', 'FTbertKM', 'contrastive']:
            X_train_remaining_inds = [x for x in X_train_original_inds if x not in X_train_current_inds
                                      and x not in X_discarded_inds]
        else:
            X_train_remaining_inds = list(np.delete(X_train_remaining_inds, sampled_ind))

        # Assert no common data in Dlab and Dpool
        assert bool(not set(X_train_current_inds) & set(X_train_remaining_inds))

        # Assert unique (no duplicate) inds in Dlab & Dpool
        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        # Assert each list of inds unique
        set(X_train_original_inds).difference(set(X_train_current_inds))

        results_per_iteration['last_iteration'] = current_iteration
        results_per_iteration['current_annotations'] = current_annotations
        results_per_iteration['annotations_per_iteration'] = acq_size
        results_per_iteration['X_val_inds'] = list(map(int, X_val_inds))

        print("\n")
        print("*" * 12)
        print("End of iteration {}:".format(current_iteration))
        if 'loss' in test_results.keys():
            print("Train loss {}, Val loss {}, Test loss {}".format(train_results['train_loss'],
                                                                    train_results['loss'],
                                                                    test_results['loss']))

        print("Annotated {} samples".format(acq_size))
        print("Current labeled (training) data: {} samples".format(len(X_train_current_inds)))
        print("Remaining budget: {} (in samples)".format(total_annotations - current_annotations))
        print("*" * 12)
        print()

        current_iteration += 1

        print('Saving json with the results....')

        with open(os.path.join(results_dir, 'results_of_iteration.json'), 'w') as f:
            json.dump(results_per_iteration, f)

        # Check budget
        if total_annotations - current_annotations < acq_size:
            annotations_per_iteration = total_annotations - current_annotations

        if annotations_per_iteration == 0:
            break
    print('The end!....')

if __name__ == '__main__':
    import argparse
    import random

    ##########################################################################
    # Setup args
    ##########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",
                        type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    ##########################################################################
    # Model args
    ##########################################################################
    parser.add_argument("--model_type", default="bert", type=str, help="Pretrained model")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str, help="Pretrained ckpt")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        default=True,
        type=bool,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true",
        default=False,
        help="Set this flag if you are using an uncased model.",
    )
    ##########################################################################
    # Training args
    ##########################################################################
    parser.add_argument("--do_train", default=True, type=bool, help="If true do train")
    parser.add_argument("--do_eval", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="If true do evaluation")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=256, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_thr", default=None, type=int, help="apply min threshold to warmup steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("-seed", "--seed", required=False, type=int, help="seed")
    parser.add_argument("-patience", "--patience", required=False, type=int, default=None, help="patience for early stopping (steps)")
    ##########################################################################
    # Data args
    ##########################################################################
    parser.add_argument("--dataset_name", default=None, required=True, type=str,
                        help="Dataset [mrpc, ag_news, qnli, sst-2]")
    # parser.add_argument("--task_name", default=None, type=str, help="Task [MRPC, AG_NEWS, QNLI, SST-2]")
    parser.add_argument("--max_seq_length", default=256, type=int, help="Max sequence length")
    # parser.add_argument("--counterfactual", default=None, type=str, help="type of counterfactual data: [orig, new, combined]")
    # parser.add_argument("--ood_name", default=None, type=str, help="name of ood dataset for counterfactual data: [amazon, yelp, semeval]")
    ##########################################################################
    # AL args
    ##########################################################################
    parser.add_argument("-acquisition", "--acquisition", required=True,
                        type=str,
                        choices=acquisition_functions,
                        help="Choose an acquisition function to be used for AL.")
    parser.add_argument("-budget", "--budget", required=False,
                        default=50, type=int,
                        help="budget \in [1,100] percent. if > 100 then it represents the total annotations")
    parser.add_argument("-mc_samples", "--mc_samples", required=False, default=None, type=int,
                        help="number of MC forward passes in calculating uncertainty estimates")
    # parser.add_argument("-variant", "--variant", required=False, default="prob", type=str,
    #                     help="prob for probabilistic, det for deterministic")
    # parser.add_argument("-sim", "--simulation", required=False,
    #                     default=False, type=bool,
    #                     help="percentage of initial training data")
    # parser.add_argument("--adaptation", required=False,
    #                     default=False, type=bool,
    #                     help="if True use ckpt from previous al iteration")
    # parser.add_argument("--adaptation_best", required=False,
    #                     default=False, type=bool,
    #                     help="if True use ckpt from previous al iteration")
    parser.add_argument("--resume", required=False,
                        default=False,
                        type=bool,
                        help="if True resume experiment")
    # parser.add_argument("--oversampling", required=False,
    #                     default=False,
    #                     type=bool,
    #                     help="if oversampling")
    # parser.add_argument("--adapt_new", required=False,
    #                     default=False,
    #                     type=bool,
    #                     help="if True load previous optimizer")
    # parser.add_argument("--adapters", required=False,
    #                     default=False,
    #                     type=bool,
    #                     help="if True load previous optimizer")
    parser.add_argument("--acquisition_size", required=False,
                        default=None,
                        type=int,
                        help="acquisition size at each AL iteration; if None we sample 1%")
    parser.add_argument("--init_train_data", required=False,
                        default=None,
                        type=int,
                        help="initial training data for AL; if None we sample 1%")
    parser.add_argument("--indicator", required=False,
                        default=None,
                        type=str,
                        help="Experiment indicator")
    parser.add_argument("--init", required=False,
                        default="random",
                        type=str,
                        help="random or alps")
    # ##########################################################################
    # # DA args
    # ##########################################################################
    # parser.add_argument("--da", default="ssmba", type=str, help="Apply DA method: [ssmba]")
    # parser.add_argument("--augm_bs", default=32, type=int, help="eval batch size for mlm that creates augmentations")
    # parser.add_argument("--noise_prob", default=0.15, type=float, help="Probability that a token will be changed "
    #                                                                    "(80% masked, 10% random, 10% unmasked)")
    # parser.add_argument("--augm_val", default=False, type=bool, help="if true augment val set")
    # parser.add_argument("--augm_test", default=False, type=bool, help="if true augment test set")
    # parser.add_argument("--add_adv", default=False, type=bool, help="if true add adv data to train set")
    # parser.add_argument("--add_all", default=False, type=bool, help="if true add adv data to train set at each iter -"
    #                                                                 " if False with replacement")
    # parser.add_argument("--add_adv_per", default=1.0, type=float, help="percentage of adv data to add randomly to train set")
    # # parser.add_argument("--da_set", default="lab", type=str, help="if lab augment labeled set else unlabeled")
    # # parser.add_argument("--num_per_augm", default=1, type=int, help="number of augmentations per example")
    # # parser.add_argument("--num_augm", default=100, type=int, help="number of examples to augment")
    # # parser.add_argument("--da_all", default=False, type=bool, help="if True augment the entire dataset")
    # parser.add_argument("--uda", default=False, type=bool, help="if true consistency loss, else supervised learning")
    # parser.add_argument("--uda_confidence_thresh", default=0.5, type=float, help="confidence threshold")
    # parser.add_argument("--uda_softmax_temp", default=0.4, type=float, help="temperature to sharpen predictions")
    # parser.add_argument("--uda_coeff", default=1, type=float, help="lambda value (weight) of KL loss")
    # parser.add_argument("--uda_per", default=2, type=int, help="number of times larger unlab from lab data")
    # parser.add_argument("--uda_max_inds", default=10000, type=int, help="max number of unlabeled data for consistency training")
    parser.add_argument("--reverse", default=False, type=bool, help="if True choose opposite data points")
    ##########################################################################
    # Contrastive acquisition args
    ##########################################################################
    parser.add_argument("--mean_embs", default=False, type=bool, help="if True use bert mean embeddings for kNN")
    parser.add_argument("--mean_out", default=False, type=bool, help="if True use bert mean outputs for kNN")
    parser.add_argument("--cls", default=False, type=bool, help="if True use cls embedding for kNN")
    # parser.add_argument("--kl_div", default=True, type=bool, help="if True choose KL divergence for scoring")
    parser.add_argument("--ce", default=False, type=bool, help="if True choose cross entropy for scoring")
    parser.add_argument("--operator", default="mean", type=str, help="operator to combine scores of neighbours")
    parser.add_argument("--num_nei", default=10, type=float, help="number of kNN to find")
    parser.add_argument("--conf_mask", default=False, type=bool, help="if True mask neighbours with confidence score")
    parser.add_argument("--conf_thresh", default=0., type=float, help="confidence threshold")
    parser.add_argument("--knn_lab", default=False, type=bool, help="if True queries are unlabeled data"
                                                                      "else labeled" )
    parser.add_argument("--bert_score", default=False, type=bool, help="if True use bertscore similarity" )
    parser.add_argument("--tfidf", default=False, type=bool, help="if True use tfidf scores" )
    parser.add_argument("--bert_rep", default=False, type=bool, help="if True use bert embs (pretrained) similarity" )
    ##########################################################################
    # Server args
    ##########################################################################
    # parser.add_argument("-g", "--gpu", required=False, default='0', help="gpu on which this experiment runs")
    # parser.add_argument("-server", "--server", required=False,default='ford', help="server on which this experiment runs")
    # parser.add_argument("--debug", required=False, default=False, help="debug mode")

    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = 0 if args.no_cuda else 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    print('device: {}'.format(args.device))

    # Setup args
    if args.seed == None:
        seed = random.randint(1, 9999)
        args.seed = seed
    if args.task_name is None: args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR

    if args.dataset_name in glue_datasets:
        args.data_dir = os.path.join(GLUE_DIR, args.task_name)
    else:
        args.data_dir = os.path.join(DATA_DIR, args.task_name)

    args.overwrite_cache = bool(True)
    args.evaluate_during_training = True

    # Output dir
    ckpt_dir = os.path.join(CKPT_DIR, '{}_{}_{}_{}'.format(args.dataset_name, args.model_type, args.acquisition, args.seed))
    args.output_dir = os.path.join(ckpt_dir, '{}_{}'.format(args.dataset_name, args.model_type))
    # if args.model_type == 'allenai/scibert': args.output_dir = os.path.join(ckpt_dir,
    #                                                                         '{}_{}'.format(args.dataset_name, 'bert'))
    # if args.acquisition is not None and args.variant is not None:
    #     if args.acquisition == 'entropy' or args.acquisition == 'least_conf':
    #         args.output_dir = os.path.join(args.output_dir,
    #                                        "{}-{}-{}".format(args.variant, args.acquisition, args.seed))
    #     else:
    #         args.output_dir = os.path.join(args.output_dir, "{}-{}".format(args.acquisition, args.seed))
    # else:
    #     args.output_dir = os.path.join(args.output_dir, "full-{}".format(args.seed))
    # if args.adaptation: args.output_dir += '-adapt'
    # if args.adaptation_best: args.output_dir += '-adapt-best'
    # if args.oversampling: args.output_dir += '-oversampling'
    # if args.adapt_new: args.output_dir += '-new'
    # if args.adapters: args.output_dir += '-adapters'
    if args.indicator is not None: args.output_dir += '-{}'.format(args.indicator)
    # if args.patience is not None: args.output_dir += '-early{}'.format(int(args.num_train_epochs))
    # if args.tapt is not None: args.output_dir += '-tapt-{}'.format(args.tapt)
    # if args.mc_samples is None and args.acquisition in ['entropy', 'least_conf']:
    #     args.output_dir += '-vanilla'
    # if args.uda: args.output_dir += '-uda-{}'.format(int(args.uda_per))
    # if args.augm_val: args.output_dir += '-augm-val'
    # if args.add_adv:
    #     args.output_dir += '-add'
    #     if args.add_all:
    #         args.output_dir += '-all'
    #     else:
    #         args.output_dir += '-rep'
    if args.reverse: args.output_dir += '-reverse'
    if args.mean_embs: args.output_dir += '-inputs'
    if args.mean_out: args.output_dir += '-outputs'
    if args.cls: args.output_dir += '-cls'
    if args.ce: args.output_dir += '-ce'
    if args.operator != "mean" and args.acquisition=="adv_train": args.output_dir += '-{}'.format(args.operator)
    if args.knn_lab: args.output_dir += '-lab'
    if args.bert_score: args.output_dir += '-bs'
    if args.bert_rep: args.output_dir += '-br'
    if args.tfidf: args.output_dir += '-tfidf'
    # if args.counterfactual is not None: args.output_dir += '-{}'.format(args.counterfactual)
    print('output_dir={}'.format(args.output_dir))
    create_dir(args.output_dir)

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    args.task_name = args.task_name.lower()

    al_loop(args)