"""
Code from https://huggingface.co/transformers/v3.1.0/examples.html
"""

import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utilities.general import create_dir
from utilities.preprocessors import processors, output_modes, convert_examples_to_features

logger = logging.getLogger(__name__)


def get_glue_dataset(args, data_dir, task, model_type, evaluate=False, test=False, contrast=False, ood=False,
                     ood_name=None):
    """
    Loads a dataset (raw text).
    :param data_dir: path ../data/[task]
    :param task: glue task  ("sst-2", "qqp", "qnli", "ag_news", "dbpedia", "pubmed")
    :param evaluate:
    :return:
    """
    create_dir(data_dir)
    processor = processors[task]()
    output_mode = output_modes[task]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )
    # Dataset
    # Load data features from cache or dataset file
    if test:
        filename = "cached_{}_{}_original".format("test_contrast",
                                                  str(task)) if contrast else "cached_{}_{}_original".format("test",
                                                                                                             str(task))
        if ood: filename += '_ood'
        cached_dataset = os.path.join(
            data_dir,
            filename
        )
    else:
        if evaluate and contrast:
            filename = "cached_{}_{}_original".format("dev_contrast", str(task))
        else:
            filename = "cached_{}_{}_original".format("dev" if evaluate else "train", str(task))
        cached_dataset = os.path.join(
            data_dir,
            filename,
        )
    if ood_name is not None: cached_dataset += '_{}'.format(ood_name)

    if os.path.exists(cached_dataset):
        logger.info("Loading dataset from cached file %s", cached_dataset)
        dataset = torch.load(cached_dataset)
    else:
        logger.info("Creating dataset from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if test:
            if ood:
                examples = (
                    processor.get_test_examples_ood(data_dir)
                )
            else:
                examples = (
                    processor.get_contrast_examples("test") if contrast else processor.get_test_examples(data_dir)
                )
        else:
            if evaluate and contrast:
                examples = (
                    processor.get_contrast_examples("dev")
                )
            else:
                examples = (
                    processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
                )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if task in ['sst-2', 'cola', 'ag_news', 'dbpedia', 'trec-6', 'imdb', 'pubmed', 'sentiment']:
            X = [x.text_a for x in examples]
        elif task in ['mrpc', 'mnli', 'qnli', 'rte', 'qqp', 'wnli', 'twitterppdb', 'nli']:
            X = list(zip([x.text_a for x in examples], [x.text_b for x in examples]))
        # elif task == 'mnli':
        #     X = list(zip([x.text_a for x in examples], [x.text_b for x in examples]))
        else:
            print(task)
            NotImplementedError
        y = [x.label for x in examples]
        dataset = [X, y]

        logger.info("Saving dataset into cached file %s", cached_dataset)
        torch.save(dataset, cached_dataset)

        # Save Tensor Dataset
        if test:
            filename = "test_contrast" if contrast else "test"
            if ood: filename += '_ood'
            features_dataset = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}_original".format(
                    # "test",
                    filename,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
        else:
            filename = "dev" if evaluate else "train"
            if contrast: filename += "_contrast"
            features_dataset = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}_original".format(
                    # "dev" if evaluate else "train",
                    filename,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
        # if args.counterfactual is not None: features_dataset += '_{}'.format(args.counterfactual)
        torch.save(features, features_dataset)

    return dataset


def get_glue_tensor_dataset(X_inds, args, task, tokenizer, train=False,
                            evaluate=False, test=False, augm=False, X_orig=None, X_augm=None, y_augm=None,
                            augm_features=None, dpool=False,
                            contrast=False, contrast_ori=False, ood=False, data_dir=None, ood_name=None,
                            counter_name=None):
    """
    Load tensor dataset (not original/raw).
    :param X_inds: list of indices to keep in the dataset (if None keep all)
    :param args:
    :param task:
    :param tokenizer:
    :param train: if True train dataset
    :param evaluate: if True dev dataset
    :param test: if True test dataset
    :param augm: if True augmented dataset
    :param X: augmented text (inputs)
    :param y: augmented labels (original if augmentation of labeled data) if unlabeled ?
    :return:
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task.lower()]()
    output_mode = output_modes[task.lower()]
    # Load data features from cache or dataset file

    if data_dir is None: data_dir = args.data_dir
    if test:
        prefix = "test"
    elif evaluate:
        prefix = "dev"
    elif train:
        prefix = "train"
    elif augm:
        prefix = "augm"
    else:
        prefix = "???"
    if contrast: prefix += "_contrast"
    if contrast_ori: prefix += "_contrast_ori"
    if ood: prefix += "_ood"
    # if args.counterfactual is not None: prefix += "_{}".format(args.counterfactual)
    if ood_name is not None: prefix += "_{}".format(ood_name)
    if counter_name is not None: prefix += "_{}".format(counter_name)

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}_original".format(
            prefix,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and data_dir == args.data_dir:  # and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if X_inds is not None:
            logger.info("Selecting subsample...")
            features = list(np.array(features)[X_inds])
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if test:
            if ood:
                examples = (processor.get_test_examples_ood(data_dir)) if ood_name is None else (
                    processor.get_test_examples_ood(data_dir, ood_name))
            else:
                examples = (processor.get_contrast_examples("test", contrast_ori) if (
                            contrast or contrast_ori) else processor.get_test_examples(data_dir))
        elif evaluate:
            examples = (
                processor.get_contrast_examples("dev") if contrast else processor.get_dev_examples(args.data_dir))
        elif train:
            examples = (processor.get_train_examples(args.data_dir))

        ################################################################
        if X_inds is not None:
            examples = list(np.array(examples)[X_inds])
            if hasattr(args, 'annotations_per_iteration') and hasattr(args, 'oversampling'):
                if args.oversampling and len(examples) != len(X_inds):
                    new_samples_inds = list(np.array(X_inds)[-args.annotations_per_iteration:])
                    examples += list(np.array(examples)[new_samples_inds])
                    assert len(examples) == len(X_inds)
        ################################################################
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

    if augm_features is not None:
        # append train + augmented features (for DA supervised learning)
        features = features + augm_features

    if augm and augm_features is None:
        # return augmented features (to later append with trainset for DA supervised learning)
        return features

    else:
        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)

        # avg_token_length = np.mean([len([i for i in t if i != 0]) for t in all_input_ids])
        # std_token_length = np.std([len([i for i in t if i != 0]) for t in all_input_ids])
        # print('\nAvg token length: {} +- {}\n'.format(avg_token_length, std_token_length))

        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        if X_inds is not None and augm_features is None:
            assert len(dataset) == len(X_inds)

        return dataset
