import collections
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric
from torch import nn
from torch.nn.functional import normalize
from tqdm import tqdm

# import faiss

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utilities.data_loader import get_glue_tensor_dataset
from utilities.preprocessors import processors
from utilities.trainers import my_evaluate

# from acquisition.bertscorer import calculate_bertscore


logger = logging.getLogger(__name__)


def contrastive_acquisition(args, annotations_per_iteration, X_original, y_original,
                            labeled_inds, candidate_inds, discarded_inds, original_inds,
                            tokenizer,
                            train_results,
                            results_dpool, logits_dpool, bert_representations=None,
                            train_dataset=None,
                            model=None,
                            tfidf_dtrain_reprs=None, tfidf_dpool_reprs=None):
    """

    :param args: arguments (such as flags, device, etc)
    :param annotations_per_iteration: acquisition size
    :param X_original: list of all data
    :param y_original: list of all labels
    :param labeled_inds: indices of current labeled/training examples
    :param candidate_inds: indices of current unlabeled examples (pool)
    :param discarded_inds: indices of examples that should not be considered for acquisition/annotation
    :param original_inds: indices of all data (this is a list of indices of the X_original list)
    :param tokenizer: tokenizer
    :param train_results: dictionary with results from training/validation phase (for logits) of training set
    :param results_dpool: dictionary with results from training/validation phase (for logits) of unlabeled set (pool)
    :param logits_dpool: logits for all examples in the pool
    :param bert_representations: representations of pretrained bert (ablation)
    :param train_dataset: the training set in the tensor format
    :param model: the fine-tuned model of the iteration
    :param tfidf_dtrain_reprs: tf-idf representations of training set (ablation)
    :param tfidf_dpool_reprs: tf-idf representations of unlabeled set (ablation)
    :return:
    """
    """
    CAL (Contrastive Active Learning)
    Acquire data by choosing those with the largest KL divergence in the predictions between a candidate dpool input
     and its nearest neighbours in the training set.
     Our proposed approach includes:
     args.cls = True
     args.operator = "mean"
     the rest are False. We use them (True) in some experiments for ablation/analysis
     args.mean_emb = False
     args.mean_out = False
     args.bert_score = False 
     args.tfidf = False 
     args.reverse = False
     args.knn_lab = False
     args.ce = False
    :return:
    """
    processor = processors[args.task_name]()
    if model is None and train_results is not None:
        model = train_results['model']

    if args.bert_score:  # BERT score representations (ablation)
        train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
        bert_score_matrix, bs_calc_time = calculate_bertscore(args, X_original, original_inds)
        assert bert_score_matrix.shape[0] == len(original_inds), 'bs {}, ori'.format(bert_score_matrix.shape[0],
                                                                                     len(original_inds))
        kl_scores = []
        distances = []
        pairs = []
        dist = DistanceMetric.get_metric('euclidean')
        num_adv = None
        for unlab_i, zipped in enumerate(zip(candidate_inds, logits_dpool)):
            candidate, unlab_logit = zipped
            all_similarities = bert_score_matrix[candidate]
            labeled_data_similarities = all_similarities[labeled_inds]
            labeled_neighborhood_inds = np.argpartition(labeled_data_similarities, -args.num_nei)[-args.num_nei:]
            distances_ = labeled_data_similarities[labeled_neighborhood_inds]
            distances.append(distances_)
            labeled_neighbours_labels = train_dataset.tensors[3][labeled_neighborhood_inds]
            neigh_prob = F.softmax(train_logits[labeled_neighborhood_inds], dim=-1)

            if args.ce:
                kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                               labeled_neighbours_labels])
            else:
                uda_softmax_temp = 1
                candidate_log_prob = F.log_softmax(unlab_logit / uda_softmax_temp, dim=-1)
                kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
            # confidence masking
            if args.conf_mask:
                conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
                conf_mask = conf_mask.type(torch.float32)
                kl = kl * conf_mask.numpy()
            if args.operator == "mean":
                kl_scores.append(kl.mean())
            elif args.operator == "max":
                kl_scores.append(kl.max())
            elif args.operator == "median":
                kl_scores.append(np.median(kl))

        distances = np.array([np.array(xi) for xi in distances])

        # # select argmax
        # selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]
        # select argmax
        if args.reverse:
            selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
        else:
            selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]

    elif args.tfidf and args.cls:  # Half neighbourhood with tfidf - half with cls embs (ablation)
        if train_dataset is None:
            train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        dtrain_tfidf = tfidf_dtrain_reprs
        dpool_tfidf = tfidf_dpool_reprs

        embs = 'bert_cls'
        dtrain_cls = normalize(_train_results[embs]).detach().cpu()
        dpool_cls = normalize(results_dpool[embs]).detach().cpu()

        distances = None
        nei_stats_list = []
        num_adv = None
        if not args.knn_lab:
            # centroids: UNLABELED data points

            # tfidf neighbourhood
            neigh_tfidf = KNeighborsClassifier(n_neighbors=args.num_nei)
            neigh_tfidf.fit(X=dtrain_tfidf, y=np.array(y_original)[labeled_inds])

            # cls neighbourhood
            neigh_cls = KNeighborsClassifier(n_neighbors=args.num_nei)
            neigh_cls.fit(X=dtrain_cls, y=np.array(y_original)[labeled_inds])

            criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
            dist = DistanceMetric.get_metric('euclidean')

            kl_scores = []
            num_adv = 0
            distances = []
            pairs = []
            label_list = processor.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}
            for unlab_i, unlab_logit in enumerate(
                    tqdm(logits_dpool, desc="Finding neighbours for every unlabeled data point")):
                # unlab candidate data point
                unlab_true_label = label_map[y_original[candidate_inds[unlab_i]]]
                unlab_pred_label = int(np.argmax(unlab_logit))
                correct_prediction = True if unlab_true_label == unlab_pred_label else False

                # tfidf neighbourhood
                unlab_tfidf = dpool_tfidf[unlab_i]
                distances_tfidf, neighbours_tfidf = neigh_tfidf.kneighbors(X=[unlab_tfidf.numpy()],
                                                                           return_distance=True)
                labeled_neighbours_tfidf_inds = np.array(labeled_inds)[neighbours_tfidf[0]]  # orig inds
                labeled_neighbours_tfidf_labels = train_dataset.tensors[3][neighbours_tfidf[0]]
                logits_neigh_tfidf = [train_logits[n] for n in neighbours_tfidf]
                preds_neigh_tfidf = [np.argmax(train_logits[n], axis=1) for n in neighbours_tfidf]
                neigh_prob_tfidf = F.softmax(train_logits[neighbours_tfidf], dim=-1)

                # cls neighbourhood
                unlab_cls = dpool_cls[unlab_i]
                distances_cls, neighbours_cls = neigh_cls.kneighbors(X=[unlab_cls.numpy()], return_distance=True)
                labeled_neighbours_cls_inds = np.array(labeled_inds)[neighbours_cls[0]]  # orig inds
                labeled_neighbours_cls_labels = train_dataset.tensors[3][neighbours_cls[0]]
                logits_neigh_cls = [train_logits[n] for n in neighbours_cls]
                preds_neigh_cls = [np.argmax(train_logits[n], axis=1) for n in neighbours_cls]
                neigh_prob_cls = F.softmax(train_logits[neighbours_cls], dim=-1)

                distances.append((distances_tfidf[0].mean(), distances_cls[0].mean()))
                common_neighbours_inds = [x for x in neighbours_tfidf[0] if x in neighbours_cls[0]]
                common_neighbours_labels = train_dataset.tensors[3][common_neighbours_inds]

                common_neighbours_inds_orig = list(np.array(labeled_inds)[common_neighbours_inds])
                common_neighbours_labels_orig = [label_map[y_original[x]] for x in common_neighbours_inds_orig]
                assert sorted(common_neighbours_labels_orig) == sorted(common_neighbours_labels)

                # same predicted label with neighbourhood (percentage)
                pred_label_tfif_per = len([x for x in labeled_neighbours_tfidf_labels if x == unlab_pred_label]) / len(
                    labeled_neighbours_tfidf_labels)
                pred_label_cls_per = len([x for x in labeled_neighbours_cls_labels if x == unlab_pred_label]) / len(
                    labeled_neighbours_cls_labels)

                # same true label with neighbourhood (percentage)
                true_label_tfif_per = len([x for x in labeled_neighbours_tfidf_labels if x == unlab_true_label]) / len(
                    labeled_neighbours_tfidf_labels)
                true_label_cls_per = len([x for x in labeled_neighbours_cls_labels if x == unlab_true_label]) / len(
                    labeled_neighbours_cls_labels)

                nei_stats = {'pred_label_tfif_per': pred_label_tfif_per,
                             'pred_label_cls_per': pred_label_cls_per,
                             'true_label_tfif_per': true_label_tfif_per,
                             'true_label_cls_per': true_label_cls_per,
                             'common_neighbours': len(common_neighbours_inds_orig)}
                nei_stats_list.append(nei_stats)

                # calculate score
                if args.ce:
                    kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                   labeled_neighbours_tfidf_labels]
                                  + [criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                     labeled_neighbours_cls_labels])
                else:
                    uda_softmax_temp = 1
                    candidate_log_prob = F.log_softmax(unlab_logit / uda_softmax_temp, dim=-1)
                    kl = np.array(
                        [torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob_tfidf]
                        + [torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob_cls])
                # confidence masking
                # if args.conf_mask:
                #     conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
                #     conf_mask = conf_mask.type(torch.float32)
                #     kl = kl * conf_mask.numpy()
                if args.operator == "mean":
                    kl_scores.append(kl.mean())
                elif args.operator == "max":
                    kl_scores.append(kl.max())
                elif args.operator == "median":
                    kl_scores.append(np.median(kl))

            # # select argmax
            # selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]
            # select argmax
            if args.reverse:
                selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
            else:
                selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]

    else:  # standard method
        if train_dataset is None:
            train_dataset = get_glue_tensor_dataset(labeled_inds, args, args.task_name, tokenizer, train=True)
        _train_results, train_logits = my_evaluate(train_dataset, args, model, prefix="",
                                                   al_test=False, mc_samples=None,
                                                   return_mean_embs=args.mean_embs,
                                                   return_mean_output=args.mean_out,
                                                   return_cls=args.cls
                                                   )
        if args.bert_rep and bert_representations is not None:
            # Use representations of pretrained model
            dtrain_reprs = bert_representations[labeled_inds]
            dpool_reprs = bert_representations[candidate_inds]
        elif tfidf_dtrain_reprs is not None:
            # Use tfidf representations
            dtrain_reprs = tfidf_dtrain_reprs
            dpool_reprs = tfidf_dpool_reprs
        else:
            # Use representations of current fine-tuned model *CAL*
            if args.mean_embs and args.cls:
                dtrain_reprs = torch.cat((_train_results['bert_mean_inputs'], _train_results['bert_cls']), dim=1)
                dpool_reprs = torch.cat((results_dpool['bert_mean_inputs'], results_dpool['bert_cls']), dim=1)
            elif args.mean_embs:
                embs = 'bert_mean_inputs'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            elif args.mean_out:
                embs = 'bert_mean_output'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            elif args.cls:
                embs = 'bert_cls'
                dtrain_reprs = _train_results[embs]
                dpool_reprs = results_dpool[embs]
            else:
                NotImplementedError

        if tfidf_dtrain_reprs is None:
            train_bert_emb = normalize(dtrain_reprs).detach().cpu()
            dpool_bert_emb = normalize(dpool_reprs).detach().cpu()
        else:
            train_bert_emb = dtrain_reprs
            dpool_bert_emb = dpool_reprs

        distances = None

        num_adv = None
        if not args.knn_lab:  # centroids: UNLABELED data points (ablation)
            #############################################################################################################################
            # Contrastive Active Learning (CAL)
            #############################################################################################################################
            neigh = KNeighborsClassifier(n_neighbors=args.num_nei)
            neigh.fit(X=train_bert_emb, y=np.array(y_original)[labeled_inds])
            # criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.BCEWithLogitsLoss()
            criterion = nn.KLDivLoss(reduction='none') if not args.ce else nn.CrossEntropyLoss()
            dist = DistanceMetric.get_metric('euclidean')

            kl_scores = []
            num_adv = 0
            distances = []
            pairs = []
            for unlab_i, candidate in enumerate(
                    tqdm(zip(dpool_bert_emb, logits_dpool), desc="Finding neighbours for every unlabeled data point")):
                # find indices of closesest "neighbours" in train set
                unlab_representation, unlab_logit = candidate
                distances_, neighbours = neigh.kneighbors(X=[candidate[0].numpy()], return_distance=True)
                distances.append(distances_[0])
                labeled_neighbours_inds = np.array(labeled_inds)[neighbours[0]]  # orig inds
                labeled_neighbours_labels = train_dataset.tensors[3][neighbours[0]]
                # calculate score
                logits_neigh = [train_logits[n] for n in neighbours]
                logits_candidate = candidate[1]
                preds_neigh = [np.argmax(train_logits[n], axis=1) for n in neighbours]
                neigh_prob = F.softmax(train_logits[neighbours], dim=-1)
                pred_candidate = [np.argmax(candidate[1])]
                num_diff_pred = len(list(set(preds_neigh).intersection(pred_candidate)))

                if num_diff_pred > 0: num_adv += 1
                if args.ce:
                    kl = np.array([criterion(unlab_logit.view(-1, args.num_classes), label.view(-1)) for label in
                                   labeled_neighbours_labels])
                else:
                    uda_softmax_temp = 1
                    candidate_log_prob = F.log_softmax(candidate[1] / uda_softmax_temp, dim=-1)
                    kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
                    # kl_scores.append(kl)
                # confidence masking
                if args.conf_mask:
                    conf_mask = torch.max(neigh_prob, dim=-1)[0] > args.conf_thresh
                    conf_mask = conf_mask.type(torch.float32)
                    kl = kl * conf_mask.numpy()
                if args.operator == "mean":
                    kl_scores.append(kl.mean())
                elif args.operator == "max":
                    kl_scores.append(kl.max())
                elif args.operator == "median":
                    kl_scores.append(np.median(kl))

            distances = np.array([np.array(xi) for xi in distances])

            logger.info('Total Different predictions for similar inputs: {}'.format(num_adv))

            # select argmax
            if args.reverse:  # if True select opposite (ablation)
                selected_inds = np.argpartition(kl_scores, annotations_per_iteration)[:annotations_per_iteration]
            else:
                selected_inds = np.argpartition(kl_scores, -annotations_per_iteration)[-annotations_per_iteration:]
            #############################################################################################################################

        else:  # centroids: LABELED data points (ablation)
            criterion = nn.KLDivLoss(reduction='sum') if not args.ce else nn.CrossEntropyLoss()
            # step 1: find neighbours for each *LABELED* data point
            N = dpool_bert_emb.shape[0]
            d = dpool_bert_emb.shape[1]
            k = 5

            xb = dpool_bert_emb.numpy()  # pool
            xq = train_bert_emb.numpy()  # candidates

            index = faiss.IndexFlatL2(d)  # build the index, d=size of vectors
            # here we assume xb contains a n-by-d numpy matrix of type float32
            index.add(xb)  # add vectors to the index
            print(index.ntotal)
            k = args.num_nei  # we want 4 similar vectors
            distances, neighbours = index.search(xq, k)
            kl_scores_per_unlab = np.array([0.] * len(candidate_inds))
            for i, pair in enumerate(neighbours):
                # labeled_logit = train_logits[i]
                labeled_prob = F.softmax(train_logits[i], dim=-1)
                # labeled_log_prob =  F.log_softmax(labeled_logit)
                # unlabeled_logits = logits_dpool[pair]
                # unlabeled_probs = [F.softmax(logits_dpool[i], dim=-1) for i in pair]
                # unlabeled_log_prob = [F.log_softmax(logits_dpool[i] / 1, dim=-1) for i in pair]
                # kl = np.array([torch.sum(criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1),
                #                                    labeled_prob), dim=-1).numpy() for n in pair])
                labeled_neighbours_label = train_dataset.tensors[3][i]
                # KL divergence for each labeled data point (candidate) and labeled (query)
                if args.ce:
                    # kl = np.array([criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1), labeled_prob).numpy()
                    #                for n in pair])
                    kl = np.array(
                        [criterion(logits_dpool[n].view(-1, args.num_classes), labeled_neighbours_label.view(-1)) for n
                         in
                         pair])
                else:
                    kl = np.array([criterion(F.log_softmax(logits_dpool[n] / 1, dim=-1), labeled_prob).numpy()
                                   for n in pair])
                # if kl socre calucalte before update with mean
                scores = np.array([np.append(kl_scores_per_unlab[pair][i], kl[i]) for i in range(0, len(pair))]).mean(
                    axis=1)
                # replace old scores for these unlabeled data
                kl_scores_per_unlab[pair] = scores

            # select argmax
            selected_inds = np.argpartition(kl_scores_per_unlab, -annotations_per_iteration)[
                            -annotations_per_iteration:]
            print()

    # map from dpool inds to original inds
    sampled_ind = list(np.array(candidate_inds)[selected_inds])  # in terms of original inds

    if num_adv is not None:
        num_adv_per = round(num_adv / len(candidate_inds), 2)

    y_lab = np.asarray(y_original, dtype='object')[labeled_inds]
    X_unlab = np.asarray(X_original, dtype='object')[candidate_inds]
    y_unlab = np.asarray(y_original, dtype='object')[candidate_inds]

    labels_list_previous = list(y_lab)
    c = collections.Counter(labels_list_previous)
    stats_list_previous = [(i, c[i] / len(labels_list_previous) * 100.0) for i in c]

    new_samples = np.asarray(X_original, dtype='object')[sampled_ind]
    new_labels = np.asarray(y_original, dtype='object')[sampled_ind]

    # Mean and std of length of selected sequences
    if args.task_name in ['sst-2', 'ag_news', 'dbpedia', 'trec-6', 'imdb', 'pubmed', 'sentiment']: # single sequence
        l = [len(x.split()) for x in new_samples]
    elif args.dataset_name in ['mrpc', 'mnli', 'qnli', 'cola', 'rte', 'qqp', 'nli']:
        l = [len(sentence[0].split()) + len(sentence[1].split()) for sentence in new_samples]  # pairs of sequences
    assert type(l) is list, "type l: {}, l: {}".format(type(l), l)
    length_mean = np.mean(l)
    length_std = np.std(l)
    length_min = np.min(l)
    length_max = np.max(l)

    # Percentages of each class
    labels_list_selected = list(np.array(y_original)[sampled_ind])
    c = collections.Counter(labels_list_selected)
    stats_list = [(i, c[i] / len(labels_list_selected) * 100.0) for i in c]

    labels_list_after = list(new_labels) + list(y_lab)
    c = collections.Counter(labels_list_after)
    stats_list_all = [(i, c[i] / len(labels_list_after) * 100.0) for i in c]

    assert len(set(sampled_ind)) == len(sampled_ind)
    assert bool(not set(sampled_ind) & set(labeled_inds))
    assert len(new_samples) == annotations_per_iteration, 'len(new_samples)={}, annotatations_per_it={}'.format(
        len(new_samples),
        annotations_per_iteration)
    assert len(labeled_inds) + len(candidate_inds) + len(discarded_inds) == len(original_inds), "labeled {}, " \
                                                                                                "candidate {}, " \
                                                                                                "disgarded {}, " \
                                                                                                "original {}".format(
        len(labeled_inds),
        len(candidate_inds),
        len(discarded_inds),
        len(original_inds))

    stats = {'length': {'mean': float(length_mean),
                        'std': float(length_std),
                        'min': float(length_min),
                        'max': float(length_max)},
             'class_selected_samples': stats_list,
             'class_samples_after': stats_list_all,
             'class_samples_before': stats_list_previous,
             }
    if distances is not None:
        if type(distances) is list:
            stats['distances'] = distances
        else:
            stats['distances'] = [float(x) for x in distances.mean(axis=1)]

    return sampled_ind, stats
