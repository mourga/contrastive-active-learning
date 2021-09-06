import collections
import json
import math
import os
import pickle
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric
from torch import nn
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from acquisition.uncertainty import select_alps
from utilities.data_loader import get_glue_dataset, get_glue_tensor_dataset
from utilities.preprocessors import output_modes
from utilities.trainers import test_transformer_model, train, train_transformer_model, my_evaluate

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sys_config import acquisition_functions, CACHE_DIR, glue_datasets, GLUE_DIR, DATA_DIR, CKPT_DIR
from utilities.general import create_dir, print_stats, create_exp_dirs

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
    X_val, y_val = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, evaluate=True)
    X_test, y_test = get_glue_dataset(args, args.data_dir, args.task_name, args.model_type, test=True)

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

    X_train_original_inds = list(np.arange(len(X_train_original)))  # original pool
    X_val_inds = list(np.arange(len(X_val)))
    X_test_inds = list(np.arange(len(X_test)))

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

    args.binary = True if len(set(np.array(y_train_original)[X_train_original_inds])) == 2 else False
    args.num_classes = len(set(np.array(y_train_original)[X_train_original_inds]))

    # if args.acquisition_size is None:
    #     args.acquisition_size = round(len(X_train_original_inds) / 100)  # 1%
    #     if args.dataset_name in ['qnli', 'ag_news']:
    #         args.acquisition_size = round(args.acquisition_size / 2)  # 0.5%
    #     # elif args.dataset_name in ['dbpedia']:
    #     #     args.acquisition_size = round(len(X_train_original_inds) / 1000)  # 0.1%
    # if args.init_train_data is None:
    #     args.init_train_data = round(len(X_train_original_inds) / 100)  # 1%
    #     if args.dataset_name in ['qnli', 'ag_news']:
    #         args.init_train_data = round(args.init_train_data / 2)  # 0.5%
    #     # elif args.dataset_name in ['dbpedia']:
    #     #     args.init_train_data = round(len(X_train_original_inds) / 1000)  # 0.1%
    #
    # if args.indicator == "small_config":
    #     args.acquisition_size = 100
    #     args.init_train_data = 100
    #     args.budget = 5100
    #
    # if args.indicator == "25_config":
    args.acquisition_size = round(len(X_train_original_inds) * 2 / 100)  # 2%
    args.init_train_data = round(len(X_train_original_inds) * 1 / 100)  # 1%
    args.budget = round(len(X_train_original_inds) * 17 / 100)  # 25%


    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    ##############################################################
    # Stats
    ##############################################################
    print()
    print_stats(np.array(y_train_original)[X_train_original_inds], 'train')
    print_stats(np.array(y_val)[X_val_inds], 'validation')
    print_stats(np.array(y_test)[X_test_inds], 'test')

    print("\nDataset for annotation: {}\nAcquisition function: {}\n"
          "Budget: {}% of labeled data\n".format(args.dataset_name,
                                                 args.acquisition,
                                                 args.budget))

    init_train_data = args.init_train_data
    init_train_percent = init_train_data / len(list(np.array(X_train_original)[X_train_original_inds])) * 100

    ##############################################################
    # Experiment dir
    ##############################################################
    results_per_iteration = {}

    results_dir = create_exp_dirs(args)
    resume_dir = results_dir

    ##############################################################
    # Get BERT representations
    ##############################################################
    bert_representations = None
    if args.bert_rep:
        if os.path.isfile(os.path.join(args.data_dir, "bert_representations.pkl")):
            print('Load bert representations...')
            with open(os.path.join(args.data_dir, "bert_representations.pkl"), 'rb') as handle:
                bert_representations = pickle.load(handle)
                assert bert_representations.shape[0] == len(X_train_original_inds)
        else:
            args.task_name = args.task_name.lower()
            args.output_mode = output_modes[args.task_name]
            ori_dataset = get_glue_tensor_dataset(X_train_original_inds, args, args.task_name, tokenizer, train=True)
            bert_config = AutoConfig.from_pretrained(
                # args.config_name if args.config_name else args.model_name_or_path,
                'bert-base-cased',
                num_labels=args.num_classes,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir,
            )
            bert_tokenizer = AutoTokenizer.from_pretrained(
                # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                'bert-base-cased',
                cache_dir=args.cache_dir,
                use_fast=args.use_fast_tokenizer,
            )
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                # args.model_name_or_path,
                'bert-base-cased',
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=bert_config,
                cache_dir=args.cache_dir,
            )
            bert_model.to(args.device)
            # eval_loss, logits, result
            _, _, _results = test_transformer_model(args, dataset=ori_dataset, model=bert_model,return_cls=True)
            bert_representations = _results["bert_cls"]
            assert bert_representations.shape[0] == len(X_train_original_inds)

            with open(os.path.join(args.data_dir, "bert_representations.pkl"), 'wb') as handle:
                pickle.dump(bert_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##############################################################
    # Get TFIDF representations
    ##############################################################
    tfidf_representations = None
    if args.tfidf:
        if os.path.isfile(os.path.join(args.data_dir, "tfidf_representations.pkl")):
            print('Load tfidf representations...')
            with open(os.path.join(args.data_dir, "tfidf_representations.pkl"), 'rb') as handle:
                tfidf_representations = pickle.load(handle)
                assert len(tfidf_representations) == len(X_train_original_inds)
        else:
            vectorizer = TfidfVectorizer(max_features=15000, lowercase=True,
                                         stop_words=feature_extraction.text.ENGLISH_STOP_WORDS)
            if type(X_train_original[0]) is list or type(X_train_original[0]) is tuple:
                vectors = vectorizer.fit_transform(
                    [s[0] + ' ' + s[1] for s in np.array(X_train_original)[X_train_original_inds]])
            else:
                vectors = vectorizer.fit_transform([s for s in np.array(X_train_original)[X_train_original_inds]])
            feature_names = vectorizer.get_feature_names()
            dense = vectors.todense()
            denselist = dense.tolist()
            tfidf_representations = denselist
            # tfidf_representations = torch.tensor(denselist)
            assert len(tfidf_representations) == len(X_train_original_inds)
            with open(os.path.join(args.data_dir, "tfidf_representations.pkl"), 'wb') as handle:
                pickle.dump(tfidf_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ##############################################################
    # Resume
    ##############################################################
    if args.resume:
        if not os.path.exists(results_dir) or not os.listdir(results_dir) or len(os.listdir(results_dir)) < 2:
            args.resume = False
            print('Experiment does not exist. Cannot resume. Start from the beginning.')
        # if os.path.exists(fig_dir) and len(os.listdir(fig_dir)) != 0:
        #     # if os.path.isfile(os.path.join(d_pool_dir, 'd_pool.json')):
        #     if os.path.isfile(os.path.join(fig_dir, 'results_of_iteration.json')):
        #         print('Experiment does exist, we found it!. Resuming...')
        #         args.resume = True
        #         resume_dir = fig_dir
        #     else:
        #         args.resume = False
        #         print('Experiment does not exist. Cannot resume. Start from the beginning.')

    if args.resume:
        print("Resume AL loop.....")
        # with open(os.path.join(results_per_iteration_dir, 'results_of_iteration.json'), 'r') as f:
        with open(os.path.join(resume_dir, 'results_of_iteration.json'), 'r') as f:
            results_per_iteration = json.load(f)
        # if os.path.isfile(os.path.join(d_pool_dir, 'd_pool.json')):
        #     with open(os.path.join(d_pool_dir, 'd_pool.json'), 'r') as f:
        #         d_pool = json.load(f)
        # else:
        #     d_pool = {}
        # with open(os.path.join(results_per_iteration_dir, 'selected_ids_per_iteration.json'), 'r') as f:
        with open(os.path.join(resume_dir, 'selected_ids_per_iteration.json'), 'r') as f:
            ids_per_it = json.load(f)

        current_iteration = results_per_iteration['last_iteration'] + 1

        X_train_current_inds = []
        for key in ids_per_it:
            X_train_current_inds += ids_per_it[key]

        X_train_remaining_inds = [i for i in X_train_original_inds if i not in X_train_current_inds]
        assert len(X_train_current_inds) + len(X_train_remaining_inds) == len(X_train_original_inds), "current {}, remaining {}, " \
                                                                                                      "original {}".format(len(X_train_current_inds),
                                                                                                                           len(X_train_remaining_inds),
                                                                                                                           len(X_train_original_inds))

        print("Current labeled dataset {}".format(len(X_train_current_inds)))
        print("Unlabeled dataset (Dpool) {}".format(len(X_train_remaining_inds)))

        if args.augm_val:
            if 'X_val_inds' in results_per_iteration.keys():
                X_val_inds = results_per_iteration['X_val_inds']
                # ori2augm_val = results_per_iteration['ori2augm_val']
        if args.add_adv:
            dpool_augm_inds = results_per_iteration['dpool_augm_inds']
        else:
            dpool_augm_inds = []
        current_annotations = results_per_iteration['current_annotations']
        annotations_per_iteration = results_per_iteration['annotations_per_iteration']
        total_annotations = round(args.budget * len(X_train_original) / 100)
        if args.budget > 100: total_annotations = args.budget
        assert current_annotations <= total_annotations, "Experiment done already!"
        total_iterations = round(total_annotations / annotations_per_iteration)

        if annotations_per_iteration != args.acquisition_size:
            annotations_per_iteration = args.acquisition_size
            print("New budget! {} more iterations.....".format(
                total_iterations - round(current_annotations / annotations_per_iteration)))

        # ##############################################################
        # # Augment Dpool
        # ##############################################################
        # if args.acquisition == "adv" or args.uda:
        #     if os.path.isfile(os.path.join(args.data_dir, "dpool_augm_{}.pkl".format(args.seed))):
        #         with open(os.path.join(args.data_dir, "dpool_augm_{}.pkl".format(args.seed)), 'rb') as handle:
        #             [X_orig_list, X_augm_list, y_augm_list, X_augm_ind_list] = pickle.load(handle)
        #     else:
        #         print("Start augmenting Dpool...")
        #         if args.add_adv:
        #             _, X_to_augment_inds, _, _ = train_test_split(X_train_original_inds,
        #                                                                                y_train_original,
        #                                                                                # train_size=init_train_percent / 100,
        #                                                                                train_size=args.init_train_data,
        #                                                                                random_state=args.seed,
        #                                                                                stratify=y_train_original)
        #         else:
        #             X_to_augment_inds = X_train_remaining_inds
        #         X_to_augment = list(np.asarray(X_train_original, dtype='object')[X_to_augment_inds])
        #         y_to_augment = list(np.asarray(y_train_original, dtype='object')[X_to_augment_inds])
        #
        #         # todo: add backtranslation
        #         X_orig_list, X_augm_list, y_augm_list, X_augm_ind_list = select_da_method(dataset=args.dataset_name,
        #                                                                                   method=args.da,
        #                                                                                   X=X_to_augment,
        #                                                                                   y=y_to_augment,
        #                                                                                   X_inds=X_to_augment_inds,
        #                                                                                   num_augm=1,
        #                                                                                   seed=args.seed,
        #                                                                                   eval_batch_size=args.augm_bs)
        #
        #         assert max(X_augm_ind_list) < len(X_train_original_inds)
        #         if not args.add_adv:
        #             assert len(X_augm_ind_list) <= len(X_train_remaining_inds), "augmented {}, remaining {}".format(len(X_augm_ind_list), len(X_train_remaining_inds))
        #
        #     if len(X_augm_ind_list) <= len(X_train_remaining_inds):
        #         X_train_remaining_inds = X_augm_ind_list
        #
        #     augm2ori = {i: o for i, o in enumerate(X_train_remaining_inds)}
        #     ori2augm = {v: k for k, v in augm2ori.items()}
        #
        #     if args.acquisition == "adv":
        #         augm2ori = {i: o for i, o in enumerate(X_augm_ind_list)}
        #         ori2augm = {v: k for k, v in augm2ori.items()}
        #
        #         new_X_orig_list = list(np.array(X_orig_list, dtype='object')[[ori2augm[x] for x in X_train_remaining_inds]])
        #         new_X_augm_list = list(np.array(X_augm_list, dtype='object')[[ori2augm[x] for x in X_train_remaining_inds]])
        #         new_X_augm_list = [x[0] for x in new_X_augm_list]
        #         new_y_augm_list = list(np.array(y_augm_list, dtype='object')[[ori2augm[x] for x in X_train_remaining_inds]])
        #
        #         dpool_augm_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, augm=True,
        #                                                      X_augm=X_augm_list, y_augm=y_augm_list, dpool=True)
        #                                                      # X_augm=new_X_augm_list, y_augm=new_y_augm_list, dpool=True)
        #         # dpool_dataset = get_glue_tensor_dataset(X_train_remaining_inds, args, args.task_name, tokenizer, train=True)
        #         dpool_dataset = get_glue_tensor_dataset(X_augm_ind_list, args, args.task_name, tokenizer, train=True)
        #
        #         if not args.add_adv:
        #             assert dpool_augm_dataset.tensors[0].size(0) == dpool_dataset.tensors[0].size(0)
        #
        #     if args.uda:
        #         uda_augm_dataset = get_glue_tensor_dataset(X_augm_ind_list, args, args.task_name, tokenizer, augm=True,
        #                                                    X_orig = X_orig_list, X_augm = X_augm_list, y_augm = y_augm_list)
        #         # uda_augm_dataset = get_glue_tensor_dataset(X_train_remaining_inds, args, args.task_name, tokenizer, augm=True,
        #         #                                        X_orig=new_X_orig_list, X_augm=new_X_augm_list, y_augm=new_y_augm_list)


        X_discarded_inds = [x for x in X_train_original_inds if x not in X_train_remaining_inds
                            and x not in X_train_current_inds]

        if args.oversampling and current_iteration != 1:
            assert len(X_train_current_inds) + len(X_train_remaining_inds) - annotations_per_iteration \
                   == len(X_train_original_inds)
        else:
            assert len(X_train_current_inds) + len(X_train_remaining_inds) + len(X_discarded_inds) == \
                   len(X_train_original_inds), "current {}, remaining {}, discarded {}, original {}".format(len(X_train_current_inds),
                                                                                               len(X_train_remaining_inds),
                                                                                               len(X_discarded_inds),
                                                                                                            len(X_train_original_inds))
        assert bool(not set(X_train_current_inds) & set(X_train_remaining_inds))

        it2per = {}  # iterations to data percentage
        val_acc_previous = None
        args.acc_best_iteration = 0
        args.acc_best = 0

        print("current iteration {}".format(current_iteration))
        print("annotations_per_iteration {}".format(annotations_per_iteration))
        print("budget {}".format(args.budget))
    else:
        ##############################################################
        # New experiment!
        ##############################################################
        ##############################################################
        # Denote labeled and unlabeled datasets
        ##############################################################
        # Pool of unlabeled data: dict containing all ids corresponding to X_train_original.
        # For each id we save (1) its true labels, (2) in which AL iteration it was selected for annotation,
        # (3) its predictive uncertainty for all iterations
        # (only for the selected ids so that we won't evaluate in the entire Dpool in every iteration)
        d_pool = {}

        # ids_per_iteration dict: contains the indices selected at each AL iteration
        ids_per_it = {}

        # ##############################################################
        # # Select validation data
        # ##############################################################
        # # for now we use the original dev set
        # al_init_prints(len(np.array(X_train_original)[X_train_original_inds]), len(np.array(X_val)[X_val_inds]),
        #                args.budget, init_train_percent)

        ##############################################################
        # Select first training data
        ##############################################################
        y_strat = np.array(y_train_original)[X_train_original_inds]

        X_train_original_after_sampling_inds = []
        X_train_original_after_sampling = []

        if args.acquisition == 'alps':
            args.init = 'alps'

        if args.init == 'random':
            X_train_init_inds, X_train_remaining_inds, _, _ = train_test_split(X_train_original_inds,
                                                                               np.array(y_train_original)[X_train_original_inds],
                                                                               # train_size=init_train_percent / 100,
                                                                               train_size=args.init_train_data,
                                                                               random_state=args.seed,
                                                                               stratify=y_strat)

        elif args.init == 'alps':
            X_train_init_inds = select_alps(args, sampled=[], acquisition_size=args.init_train_data,
                                            original_inds=X_train_original_inds)
            X_train_remaining_inds = [x for x in X_train_original_inds if x not in X_train_init_inds]
        else:
            print(args.init)
            raise NotImplementedError

        ####################################################################
        # Create Dpool and Dlabels
        ####################################################################
        X_train_init = list(np.asarray(X_train_original, dtype='object')[X_train_init_inds])
        y_train_init = list(np.asarray(y_train_original, dtype='object')[X_train_init_inds])

        for i in list(set(y_train_init)):
            init_train_dist_class = 100 * np.sum(np.array(y_train_init) == i) / len(y_train_init)
            print('init % class {}: {}'.format(i, init_train_dist_class))

        if X_train_original_after_sampling_inds == []:
            assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(X_train_original_inds), 'init {}, remaining {}, original {}'.format(len(X_train_init_inds),
                                                                                                                                                   len(X_train_remaining_inds),
                                                                                                                                                   len(X_train_original_inds))
        else:
            assert len(X_train_init_inds) + len(X_train_remaining_inds) == len(X_train_original_after_sampling_inds)

        # d_pool.update({str(key): {'label': int(y_train_original[key]), 'it_selected': None, 'iterations': {}} for key in
        d_pool.update({str(key): {'label': y_train_original[key], 'it_selected': None, 'iterations': {}} for key in
                       X_train_original_inds})
        for key in X_train_init_inds:
            d_pool[str(key)]['it_selected'] = 0  # 0 means it was selected in the initial training set (randomly)

        ids_per_it.update({str(0): list(map(int, X_train_init_inds))})
        assert len(ids_per_it[str(0)]) == args.init_train_data

        ####################################################################
        # Annotations & budget
        ####################################################################
        current_annotations = len(X_train_init)  # without validation data
        if X_train_original_after_sampling == []:
            # total_annotations = round(args.budget * len(X_train_original) / 100)
            total_annotations = round(args.budget * len(np.array(X_train_original)[X_train_original_inds]) / 100)
        else:
            total_annotations = round(args.budget * len(X_train_original_after_sampling) / 100)
        if args.budget > 100: total_annotations = args.budget
        annotations_per_iteration = args.acquisition_size
        total_iterations = math.ceil(total_annotations / annotations_per_iteration)

        X_train_current_inds = X_train_init_inds.copy()

        # ##############################################################
        # # Augment Dpool
        # ##############################################################
        # dpool_augm_inds = []
        # if args.acquisition == "adv" or args.uda:
        #
        #     if os.path.isfile(os.path.join(args.data_dir, "dpool_augm_{}.pkl".format(args.seed))):
        #         print('Load augmented dpool...')
        #         with open(os.path.join(args.data_dir, "dpool_augm_{}.pkl".format(args.seed)), 'rb') as handle:
        #             [X_orig_list, X_augm_list, y_augm_list, X_augm_ind_list] = pickle.load(handle)
        #     else:
        #         print('Augmented dpool...')
        #         X_to_augment_inds = X_train_remaining_inds
        #         X_to_augment = list(np.asarray(X_train_original, dtype='object')[X_train_remaining_inds])
        #         y_to_augment = list(np.asarray(y_train_original, dtype='object')[X_train_remaining_inds])
        #
        #         # todo: add backtranslation
        #         X_orig_list, X_augm_list, y_augm_list, X_augm_ind_list = select_da_method(dataset=args.dataset_name,
        #                                                                                   method=args.da,
        #                                                                                   X=X_to_augment,
        #                                                                                   y=y_to_augment,
        #                                                                                   X_inds=X_to_augment_inds,
        #                                                                                   num_augm=1,
        #                                                                                   seed=args.seed,
        #                                                                                   eval_batch_size=args.augm_bs)
        #
        #         assert max(X_augm_ind_list) < len(X_train_original_inds)
        #         with open(os.path.join(args.data_dir, "dpool_augm_{}.pkl".format(args.seed)), 'wb') as handle:
        #             pickle.dump([X_orig_list, X_augm_list, y_augm_list, X_augm_ind_list],
        #                         handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        #     # if args.acquisition == "adv":
        #     if len(X_augm_ind_list) != len(X_train_remaining_inds):
        #         # X_train_remaining_inds = [augm2ori[i] for i in X_augm_ind_list]
        #         X_train_remaining_inds = [x for x in X_augm_ind_list if x not in X_train_current_inds]
        #
        #         # X_augm_original_inds = list(np.arange(len(X_augm_list)))
        #     augm2ori = {i: o for i, o in enumerate(X_train_remaining_inds)}
        #     ori2augm = {v: k for k, v in augm2ori.items()}
        #
        #     if args.acquisition == "adv":
        #         dpool_augm_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, augm=True,
        #                                                      X_augm=X_augm_list, y_augm=y_augm_list, dpool=True)
        #         dpool_dataset = get_glue_tensor_dataset(X_train_remaining_inds, args, args.task_name, tokenizer, train=True)
        #         print()
        #         assert dpool_augm_dataset.tensors[0].size(0) == dpool_dataset.tensors[0].size(0), 'augm {}, dpool {}'.format(dpool_augm_dataset.tensors[0].size(0),
        #                                                                                                                      dpool_dataset.tensors[
        #                                                                                                                          0].size(
        #                                                                                                                          0))
        #
        #     if args.uda:
        #         uda_augm_dataset = get_glue_tensor_dataset(X_augm_ind_list, args, args.task_name, tokenizer, augm=True,
        #                                                X_orig=X_orig_list, X_augm=X_augm_list, y_augm=y_augm_list)

        X_discarded_inds = [x for x in X_train_original_inds if x not in X_train_remaining_inds
                            and x not in X_train_current_inds]

        it2per = {}  # iterations to data percentage
        val_acc_previous = None
        args.acc_best_iteration = 0
        args.acc_best = 0
        current_iteration = 1

    # Assertions
    # if args.indicator is not None:
    #     if 'simulation' in args.indicator:
    #         X_train_remaining_inds = list(
    #             set(list(X_train_original_after_sampling_inds)).difference(set(X_train_current_inds)))
    #         total_annotations = round(args.budget * len(X_train_original_after_sampling_inds) / 100)
    #         total_iterations = round(total_annotations / annotations_per_iteration)
    #         assert len(X_train_remaining_inds) + len(X_train_current_inds) == len(X_train_original_after_sampling_inds)
    # else:
    #     assert len(X_train_remaining_inds) + len(X_train_current_inds) + len(X_discarded_inds) == len(X_train_original_inds)
    print()
    assert bool(not set(X_train_remaining_inds) & set(X_train_current_inds))

    """
        Indices of X_train_original: X_train_init_inds - inds of first training set (iteration 1)
                                     X_train_current_inds - inds of labeled dataset (iteration i)
                                     X_train_remaining_inds - inds of unlabeled dataset (iteration i)
                                     X_train_original_inds - inds of (full) original training set
                                     X_disgarded_inds - inds from dpool that are disgarded

    """
    # _uda_augm_dataset = uda_augm_dataset
    # adversarial_val_inds = None

    #############
    # Start AL!
    #############
    while current_iteration < total_iterations + 1:

        it2per[str(current_iteration)] = round(len(X_train_current_inds) / len(X_train_original_inds), 2) * 100

        ##############################################################
        # Train model on training dataset (Dtrain)
        ##############################################################
        print("\n Start Training model of iteration {}!\n".format(current_iteration))
        train_results = train_transformer_model(args, X_train_current_inds,
                                                X_val_inds,
                                                iteration=current_iteration,
                                                val_acc_previous=val_acc_previous,
                                                )

        val_acc_previous = train_results['acc']
        # adversarial_val_inds = train_results['val_adv_inds']
        print("\nDone Training!\n")

        ##############################################################
        # Test model on test data (D_test)
        ##############################################################
        print("\nStart Testing on test set!\n")
        # if args.dataset_name == 'sentiment':
        #     test_results = {}
        #     for name in ["new", "orig", "combined"]:
        #         test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True,
        #                                                counter_name=name)
        #         _test_results, test_logits = my_evaluate(test_dataset, args, train_results['model'], prefix="",
        #                                                  al_test=False, mc_samples=None)
        #         _test_results.pop('gold_labels', None)
        #         test_results[name] = _test_results
        # else:
        test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True)
        test_results, test_logits = my_evaluate(test_dataset, args, train_results['model'], prefix="",
                                                al_test=False, mc_samples=None)
        test_results.pop('gold_labels', None)

        ##############################################################
        # Test model on OOD test data (D_ood)
        ##############################################################
        print("\nEvaluating robustness! Start testing on OOD test set!\n")
        # if False:
        if X_test_ood is not None:
            # if args.dataset_name == 'sentiment':
            #     ood_test_results = {}
            #     for ood_name in ["amazon", "yelp", "semeval"]:
            #         ood_test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True,
            #                                                    ood=True, ood_name=ood_name)
            #         _ood_test_results, ood_test_logits = my_evaluate(ood_test_dataset, args, train_results['model'],
            #                                                          prefix="",
            #                                                          al_test=False, mc_samples=None)
            #         _ood_test_results.pop('gold_labels', None)
            #         ood_test_results[ood_name] = _ood_test_results
            #
            # else:
            if args.dataset_name == 'sst-2':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'imdb', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'IMDB'))
            elif args.dataset_name == 'imdb':
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'sst-2', tokenizer, test=True,
                                                           data_dir=os.path.join(GLUE_DIR, 'SST-2'))
            elif args.dataset_name == 'qqp':
                # ood_test_dataset = get_glue_tensor_dataset(None, args, 'mrpc', tokenizer, test=True, data_dir=os.path.join(GLUE_DIR, 'MRPC'))
                ood_test_dataset = get_glue_tensor_dataset(None, args, 'twitterppdb', tokenizer, test=True,
                                                           data_dir=os.path.join(DATA_DIR, 'TwitterPPDB'))
                # elif args.dataset_name == 'mrpc':
                #     ood_test_dataset = get_glue_tensor_dataset(None, args, 'qqp', tokenizer, test=True, data_dir=os.path.join(GLUE_DIR, 'QQP'))
                # elif args.dataset_name == 'qnli':
                #     ood_test_dataset = get_glue_tensor_dataset(None, args, 'wnli', tokenizer, test=True, data_dir=os.path.join(GLUE_DIR, 'WNLI'))
            else:
                ood_test_dataset = get_glue_tensor_dataset(None, args, args.task_name, tokenizer, test=True,
                                                           ood=True)
            ood_test_results, ood_test_logits = my_evaluate(ood_test_dataset, args, train_results['model'],
                                                            prefix="",
                                                            al_test=False, mc_samples=None)
            ood_test_results.pop('gold_labels', None)

        # ##############################################################
        # # Test model on contrast + original test data (D_test_contrast)
        # ##############################################################
        # print("\nEvaluating contrast set!\n")
        # # if False:
        # if args.dataset_name == 'imdb' and os.path.exists(IMDB_CONTR_DATA_DIR):
        #     contrast_results = contrast_acc_imdb(args, tokenizer, train_results, results_per_iteration_dir,
        #                                          iteration=current_iteration)
        # else:
        #     contrast_results = None

        ##############################################################
        # Test model on unlabeled data (Dpool)
        ##############################################################
        print("\nEvaluating Dpool!\n")
        start = time.time()
        dpool_loss, logits_dpool, results_dpool = [], [], []
        if args.acquisition not in ['random', 'alps', 'badge', 'FTbertKM']:
            # return_mean_embs = True if args.acquisition == "adv_train" and args.mean_embs else False
            # return_mean_output = True if args.acquisition == "adv_train" and args.mean_out else False
            # return_cls = True if args.acquisition == "adv_train" and args.cls else False
            dpool_loss, logits_dpool, results_dpool = test_transformer_model(args, X_train_remaining_inds,
                                                                             model=train_results['model'],
                                                                             return_mean_embs=args.mean_embs,
                                                                             return_mean_output=args.mean_out,
                                                                             return_cls=args.cls)
            results_dpool.pop('gold_labels', None)
        end = time.time()
        inference_time = end - start

        # ##############################################################
        # # Test model on augmented unlabeled data (~Dpool)
        # ##############################################################
        # if args.acquisition == "adv":  # assert indices.shape[0] == dpool_augm_dataset.tensors[0].size(0)
        #     # filter augm dpool
        #     indices = torch.tensor([ori2augm[x] for x in X_train_remaining_inds])
        #     _dpool_tensors = [torch.index_select(dpool_augm_dataset.tensors[i], 0, indices) for i
        #                       in range(0, len(dpool_augm_dataset[0]))]
        #     _dpool_augm = TensorDataset(*_dpool_tensors)
        #     aug_dpool_loss, aug_logits_dpool, aug_results_dpool = test_transformer_model(args,
        #                                                                                  X_train_remaining_inds,
        #                                                                                  model=train_results['model'],
        #                                                                                  augm_dataset=_dpool_augm)
        #     aug_results_dpool.pop('gold_labels', None)

        # ########################################################################################################
        # # compute inference on the other selected input samples until this iteration (part of training set)
        # ########################################################################################################
        # X_rest_inds = []
        # for i in range(0, current_iteration):
        #     X_rest_inds += ids_per_it[str(i)]
        # # Assert no common data in Dlab and Dpool
        # assert bool(not set(X_train_remaining_inds) & set(X_rest_inds))
        #
        # logits_rest_dpool, results_rest_dpool = [], []
        # # if args.acquisition not in ['random', 'alps']:
        # # todo: removed it to run faster experiments
        # if False:
        #     _, logits_rest_dpool, results_rest_dpool = test_transformer_model(args, X_rest_inds,
        #                                                                       model=train_results['model'])

        ##############################################################
        # Select unlabeled samples for annotation
        # -> annotate
        # -> update training dataset & unlabeled dataset
        ##############################################################
        # I moved this in the end!
        # if total_annotations - current_annotations < annotations_per_iteration:
        #     annotations_per_iteration = total_annotations - current_annotations
        #
        # if annotations_per_iteration == 0:
        #     break

        assert len(set(X_train_current_inds)) == len(X_train_current_inds)
        assert len(set(X_train_remaining_inds)) == len(X_train_remaining_inds)

        start = time.time()
        if args.acquisition == "adv":
            sampled_ind, stats, dpool_augm_inds = adv_acq_fun(args=args,
                                                              annotations_per_iteration=annotations_per_iteration,
                                                              logits_dpool=logits_dpool,
                                                              aug_logits_dpool=aug_logits_dpool,
                                                              augm2ori=augm2ori,
                                                              X_original=X_train_original,
                                                              y_original=y_train_original,
                                                              candidate_inds=X_train_remaining_inds,
                                                              labeled_inds=X_train_current_inds,
                                                              discarded_inds=X_discarded_inds,
                                                              original_inds=X_train_original_inds,
                                                              dpool_augm_inds=dpool_augm_inds)
        elif args.acquisition == "adv_train":
            if args.tfidf:
                tfidf_dtrain_reprs = torch.tensor(list(np.array(tfidf_representations)[X_train_current_inds]))
                # tfidf_dtrain_reprs = torch.tensor(tfidf_representations[X_train_current_inds])
                tfidf_dpool_reprs = torch.tensor(list(np.array(tfidf_representations)[X_train_remaining_inds]))
                # tfidf_dpool_reprs = torch.tensor(tfidf_representations[X_train_remaining_inds])
            else:
                tfidf_dtrain_reprs = None
                tfidf_dpool_reprs = None
            sampled_ind, stats = adv_train_acq_fun(args=args,
                                                   annotations_per_iteration=annotations_per_iteration,
                                                   X_original=X_train_original,
                                                   y_original=y_train_original,
                                                   labeled_inds=X_train_current_inds,
                                                   candidate_inds=X_train_remaining_inds,
                                                   discarded_inds=X_discarded_inds,
                                                   original_inds=X_train_original_inds,
                                                   tokenizer=tokenizer,
                                                   train_results=train_results,
                                                   results_dpool=results_dpool,
                                                   logits_dpool=logits_dpool,
                                                   bert_representations=bert_representations,
                                                   tfidf_dtrain_reprs=tfidf_dtrain_reprs,
                                                   tfidf_dpool_reprs=tfidf_dpool_reprs)
        else:
            sampled_ind, stats = calculate_uncertainty(args=args,
                                                       method=args.acquisition,
                                                       logits=logits_dpool,
                                                       annotations_per_it=annotations_per_iteration,
                                                       device=args.device,
                                                       iteration=current_iteration,
                                                       task=args.task_name,
                                                       oversampling=args.oversampling,
                                                       representations=None,
                                                       candidate_inds=X_train_remaining_inds,
                                                       labeled_inds=X_train_current_inds,
                                                       discarded_inds=X_discarded_inds,
                                                       original_inds=X_train_original_inds,
                                                       model=train_results['model'],
                                                       X_original=X_train_original,
                                                       y_original=y_train_original)
        end = time.time()
        selection_time = end - start

        # Update results dict
        results_per_iteration[str(current_iteration)] = {'data_percent': it2per[str(current_iteration)],
                                                         'total_train_samples': len(X_train_current_inds),
                                                         'inference_time': inference_time,
                                                         'selection_time': selection_time}
        results_per_iteration[str(current_iteration)]['val_results'] = train_results
        results_per_iteration[str(current_iteration)]['test_results'] = test_results

        if X_test_ood is not None and args.indicator == '25_config':
            results_per_iteration[str(current_iteration)]['ood_test_results'] = ood_test_results
            results_per_iteration[str(current_iteration)]['ood_test_results'].pop('model', None)

        if contrast_results is not None:
            results_per_iteration[str(current_iteration)]['contrast_test_results'] = contrast_results

        results_per_iteration[str(current_iteration)]['val_results'].pop('model', None)
        results_per_iteration[str(current_iteration)]['test_results'].pop('model', None)
        results_per_iteration[str(current_iteration)].update(stats)

        current_annotations += annotations_per_iteration

        # X_train_current_inds and X_train_remaining_inds are lists of indices of the original dataset
        # sampled_inds is a list of indices OF THE X_train_remaining_inds(!!!!) LIST THAT SHOULD BE REMOVED
        # INCEPTION %&#!@***CAUTION***%&#!@
        if args.oversampling:
            if current_iteration != 1:
                # remove previous duplicates
                _X_train_current_inds = X_train_current_inds[:-annotations_per_iteration]
                X_train_current_inds = _X_train_current_inds
            # duplicate new samples
            X_train_current_inds += list(np.array(X_train_remaining_inds)[sampled_ind])
            X_train_current_inds += list(np.array(X_train_remaining_inds)[sampled_ind])

        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
            X_train_current_inds += list(sampled_ind)
        else:
            X_train_current_inds += list(np.array(X_train_remaining_inds)[sampled_ind])

        assert len(ids_per_it[str(0)]) == args.init_train_data

        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
            selected_dataset_ids = sampled_ind
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data
        else:
            selected_dataset_ids = list(np.array(X_train_remaining_inds)[sampled_ind])
            selected_dataset_ids = list(map(int, selected_dataset_ids))  # for json
            assert len(ids_per_it[str(0)]) == args.init_train_data

        # Update d_pool & ids_per_it
        # if results_dpool != []:
        if False:
            for i, j in enumerate(X_train_remaining_inds):
                # i = counter, j = item
                if j in selected_dataset_ids:
                    d_pool[str(j)]['it_selected'] = current_iteration
                d_pool[str(j)]['iterations'].update({current_iteration: results_dpool['prob'][i]})
            for i, j in enumerate(X_rest_inds):
                d_pool[str(j)]['iterations'].update({current_iteration: results_rest_dpool['prob'][i]})
        else:
            d_pool = {}

        ids_per_it.update({str(current_iteration): selected_dataset_ids})

        assert len(ids_per_it[str(0)]) == args.init_train_data
        assert len(ids_per_it[str(current_iteration)]) == annotations_per_iteration

        if args.acquisition in ['alps', 'badge', 'adv', 'FTbertKM', 'adv_train']:
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
        if args.indicator is None and args.indicator != "small_config":
            assert set(X_train_original_inds).difference(set(X_train_current_inds)) == set(
                X_train_remaining_inds + X_discarded_inds)

        results_per_iteration['last_iteration'] = current_iteration
        results_per_iteration['current_annotations'] = current_annotations
        results_per_iteration['annotations_per_iteration'] = annotations_per_iteration
        results_per_iteration['X_val_inds'] = list(map(int, X_val_inds))
        # results_per_iteration['ori2augm_val'] = ori2augm_val
        if args.add_adv:
            results_per_iteration['dpool_augm_inds'] = list(map(int, dpool_augm_inds))
        print("\n")
        print("*" * 12)
        print("End of iteration {}:".format(current_iteration))
        if 'loss' in test_results.keys():
            print("Train loss {}, Val loss {}, Test loss {}".format(train_results['train_loss'], train_results['loss'],
                                                                    test_results['loss']))
        if args.augm_val:
            print("Num of adversarial val examples: {}".format(len(train_results['val_adv_inds'])))
        if args.acquisition == 'adv':
            print("Num of adversarial dpool examples: {} ({}%)".format(stats['num_adv'], stats['num_adv_per']))
        print("Annotated {} samples".format(annotations_per_iteration))
        print("Current labeled (training) data: {} samples".format(len(X_train_current_inds)))
        print("Remaining budget: {} (in samples)".format(total_annotations - current_annotations))
        print("*" * 12)
        print()

        current_iteration += 1

        print('Saving json with the results....')

        with open(os.path.join(results_per_iteration_dir, 'results_of_iteration.json'), 'w') as f:
            json.dump(results_per_iteration, f)
        with open(os.path.join(d_pool_dir, 'd_pool.json'), 'w') as f:
            json.dump(d_pool, f)
        with open(os.path.join(results_per_iteration_dir, 'selected_ids_per_iteration.json'), 'w') as f:
            json.dump(ids_per_it, f)

        # Check budget
        if total_annotations - current_annotations < annotations_per_iteration:
            annotations_per_iteration = total_annotations - current_annotations

        if annotations_per_iteration == 0:
            break
    print('The end!....')

    return

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
    args.task_name = args.dataset_name.upper()

    args.cache_dir = CACHE_DIR

    # if args.dataset_name in glue_datasets:
    #     args.data_dir = os.path.join(GLUE_DIR, args.task_name)
    # else:
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