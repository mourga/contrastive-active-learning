"""
This script contains code from
- https://github.com/SanghunYun/UDA_pytorch
- https://huggingface.co/transformers/v3.1.0/examples.html
"""
import glob
import json
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset, RandomSampler, DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from transformers.data.metrics import simple_accuracy

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sys_config import EXP_DIR
from utilities.data_loader import get_glue_tensor_dataset
from utilities.early_stopping import EarlyStopping
from utilities.general import create_dir, number_h
from utilities.metrics import uncertainty_metrics, compute_metrics
from utilities.preprocessors import output_modes, processors, convert_examples_to_features

logger = logging.getLogger(__name__)


def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def torch_device_one():
    return torch.tensor(1.).to(_get_device())


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x


def get_loss(args, model, sup_batch, unsup_batch, global_step, sup_criterion, unsup_criterion):
    # logits -> prob(softmax) -> log_prob(log_softmax)

    # batch
    input_ids, segment_ids, input_mask, label_ids = sup_batch
    if unsup_batch:
        ori_input_ids, ori_segment_ids, ori_input_mask, \
        aug_input_ids, aug_segment_ids, aug_input_mask, ori_labels_ids = unsup_batch

        input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
        segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
        input_mask = torch.cat((input_mask, aug_input_mask), dim=0)

    # logits
    logits = model(input_ids, segment_ids, input_mask)
    logits = logits[0]
    # sup loss
    sup_size = label_ids.shape[0]
    sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
    sup_loss = torch.mean(sup_loss)

    # unsup loss
    if unsup_batch:
        # ori
        with torch.no_grad():
            ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
            ori_logits = ori_logits[0]
            ori_prob = F.softmax(ori_logits, dim=-1)  # KLdiv target
            # ori_log_prob = F.log_softmax(ori_logits, dim=-1)

            # confidence-based masking
            # args.uda_confidence_thresh = -1
            # args.uda_softmax_temp = -1
            # args.uda_coeff = 1

            if args.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > args.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(_get_device())

        # aug
        # softmax temperature controlling
        uda_softmax_temp = args.uda_softmax_temp if args.uda_softmax_temp > 0 else 1.
        aug_log_prob = F.log_softmax(logits[sup_size:] / uda_softmax_temp, dim=-1)

        # KLdiv loss
        """
            nn.KLDivLoss (kl_div)
            input : log_prob (log_softmax)
            target : prob    (softmax)
            https://pytorch.org/docs/stable/nn.html

            unsup_loss is divied by number of unsup_loss_mask
            it is different from the google UDA official
            The official unsup_loss is divided by total
            https://github.com/google-research/uda/blob/master/text/uda.py#L175
        """
        unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                 torch_device_one())
        final_loss = sup_loss + args.uda_coeff * unsup_loss

        return final_loss, sup_loss, unsup_loss


def train(args, train_dataset, eval_dataset, model, tokenizer, unsup_dataset=None):
    """ Train the model """
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    sup_criterion = nn.CrossEntropyLoss(reduction='none')
    sup_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    sup_loader = DataLoader(train_dataset, sampler=sup_sampler, batch_size=args.train_batch_size)
    data_loader = sup_loader

    if args.max_steps > 0:
        t_total = args.max_steps
        # args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        args.num_train_epochs = args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
    else:
        # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        t_total = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.total_steps = t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    if unsup_dataset is not None:
        logger.info("  Num unlabeled examples = %d", len(unsup_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    best_val_loss, best_val_acc = 100000000.0, 0.0
    if hasattr(args, 'patience'):
        if args.patience is not None:
            early_stopping = EarlyStopping("min", args.patience)
    break_loop = False
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    # set_seed(args)  # Added here for reproducibility - we added it in the beggining of main script
    for _ in train_iterator:
        epoch_iterator = tqdm(sup_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            epoch_iterator.set_description('loss=%5.3f' % (loss.item()))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 \
                        or global_step == t_total:
                    logs = {}
                    # Evaluate
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, original=True, eval_dataset=eval_dataset)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        # for classification
                        if 'acc' in results.keys():
                            if results['loss'] < best_val_loss:
                                best_val_acc = results['acc']
                                best_val_loss = results['loss']

                                if args.current_output_dir is not None:
                                    output_dir = args.current_output_dir
                                else:
                                    output_dir = args.output_dir
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)

                                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                                logger.info("Saving model checkpoint to %s", output_dir)

                                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                                logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        if hasattr(args, 'patience'):
                            if args.patience is not None:
                                # print('patience: {}'.format(early_stopping.patience))
                                if early_stopping.stop(current_metric=results['loss']):
                                    print("Early Stopping...")
                                    break_loop = True
                                    break

                    assert args.logging_steps != 0
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["train_loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.device.type == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            if break_loop: break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if break_loop: break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step, float(best_val_acc), best_val_loss


def evaluate(args, model, tokenizer, prefix="", original=False, eval_dataset=None):
    """
    Evaluate transformer model with standard eval dataset.
    :param args:
    :param model:
    :param tokenizer:
    :param prefix:
    :return:
    """
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if eval_dataset is None:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, original=original)
        assert len(eval_dataset) != 0
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results['mnli_mm_acc'] = result['acc']
            results['mnli_mm_loss'] = eval_loss
        else:
            results.update(result)

            result['loss'] = eval_loss
            results['loss'] = eval_loss
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, overwrite_cache=False, original=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.indicator == 'alps':
        from utilities.transformers.alps_processors import alps_processors, alps_output_modes
        processors = alps_processors
        output_modes = alps_output_modes
    else:
        from utilities.general_preprocessors import processors, output_modes

    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if evaluate and original:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}_original".format(
                "dev" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
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
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def my_evaluate(eval_dataset, args, model, prefix="", al_test=False, mc_samples=None,
                return_mean_embs=False, return_mean_output=False, return_cls=False):
    """
    Evaluate model using 'eval_dataset'.
    :param eval_dataset: tensor dataset
    :param args:
    :param model:
    :param prefix: -
    :param al_test: if True then eval_dataset is Dpool
    :param mc_samples: if not None, int with number of MC forward samples
    :return:
    """
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_task_names = (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Sequential sampler - crucial!!!
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        bert_mean_output_list = None
        bert_mean_input_list = None
        bert_cls_list = None

        if mc_samples is not None and al_test:
            # Evaluation of Dpool - MC dropout
            test_losses = []
            logits_list = []
            for i in range(1, mc_samples + 1):
                test_losses_mc = []
                logits_mc = None
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    model.train()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                        if args.model_type != "distilbert":
                            inputs["token_type_ids"] = (
                                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                        # outputs = model(**inputs)
                        # if args.acquisition == 'adv_train':
                        #     outputs = model(**inputs,return_hidden=True)
                        # else:
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]

                        eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        out_label_ids = inputs["labels"].detach().cpu().numpy()
                        logits_mc = logits
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                        logits_mc = torch.cat((logits_mc, logits), 0)
                    test_losses_mc.append(eval_loss / nb_eval_steps)
                    # logits_mc.append(logits)

                test_losses.append(test_losses_mc)
                logits_list.append(logits_mc)
                preds = None

            eval_loss = np.mean(test_losses)
            logits = logits_list
            preds = torch.mean(torch.stack(logits), 0).detach().cpu().numpy()
        else:
            # Evaluation of Dval - or just no MC dropout
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs)
                    labels = inputs.pop("labels", None)
                    if hasattr(args, 'acquisition') or return_cls:
                        if return_cls or return_mean_output:
                            bert_output, bert_output_cls = model.bert(**inputs)
                            bert_mean_output = torch.mean(bert_output, dim=1)
                        if return_mean_embs:
                            bert_input = model.bert.embeddings(inputs['input_ids'])
                            bert_mean_input = torch.mean(bert_input, dim=1)

                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    # out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                    if return_mean_embs:
                        bert_mean_input_list = bert_mean_input
                    if return_mean_output:
                        bert_mean_output_list = bert_mean_output
                    if return_cls:
                        bert_cls_list = bert_output_cls
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    # out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                    # if args.acquisition == "adv_train" and return_bert_embs:
                    #     bert_output_list = torch.cat((bert_output_list, bert_output),0)

                    if return_mean_embs:
                        bert_mean_input_list = torch.cat((bert_mean_input_list, bert_mean_input), 0)
                    if return_mean_output:
                        bert_mean_output_list = torch.cat((bert_mean_output_list, bert_mean_output), 0)
                    if return_cls:
                        bert_cls_list = torch.cat((bert_cls_list, bert_output_cls), 0)

            eval_loss = eval_loss / nb_eval_steps
            logits = torch.tensor(preds)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)

            accuracy = round(simple_accuracy(out_label_ids, preds), 4)
            f1 = round(f1_score(out_label_ids, preds, average='macro'), 4)
            precision = round(precision_score(out_label_ids, preds, average='macro'), 4)
            recall = round(recall_score(out_label_ids, preds, average='macro'), 4)

            # calibration scores
            calibration_scores = uncertainty_metrics(logits, out_label_ids, pool=al_test, num_classes=args.num_classes)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            calibration_scores = {}
            accuracy, f1, precision, recall = 0., 0., 0., 0.,
            calibration_scores = None
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update({'f1_macro': f1, 'recall': recall, 'precision': precision, 'loss': eval_loss})
        results.update(calibration_scores)
        results.update({'bert_mean_inputs': bert_mean_input_list})
        results.update({'bert_mean_output': bert_mean_output_list})
        results.update({'bert_cls': bert_cls_list})
        results.update({'gold_labels': out_label_ids.tolist()})
    # return results, eval_loss, accuracy, f1, precision, recall, logits
    return results, logits


def train_transformer(args, train_dataset, eval_dataset, model, tokenizer, unsup_dataset=None,
                      val_augm_dataset=None,
                      X_val_inds=None, ori2augm_val=None):
    """
    Train a transformer model.
    :param args:
    :param train_dataset:
    :param eval_dataset:
    :param model:
    :param tokenizer:
    :return:
    """
    # 10% warmup
    args.warmup_steps = int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs / 10)
    if hasattr(args, "warmup_thr"):
        if args.warmup_thr is not None:
            args.warmup_steps = min(
                int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs / 10), args.warmup_thr)

    print("warmup steps: {}".format(args.warmup_steps))
    print("total steps: {}".format(int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs)))
    print("logging steps: {}".format(args.logging_steps))

    ##############################
    # Train model
    ##############################
    global_step, tr_loss, val_acc, val_loss = train(args, train_dataset, eval_dataset, model, tokenizer, unsup_dataset)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.current_output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)

            result, logits = my_evaluate(eval_dataset, args, model, prefix=prefix)

    eval_loss = val_loss
    return model, tr_loss, eval_loss, result


def train_transformer_model(args, X_inds, X_val_inds=None,
                            iteration=None, val_acc_previous=None, uda_augm_dataset=None,
                            eval_dataset=None,
                            adversarial_val_inds=None,
                            val_augm_dataset=None, test_augm_dataset=None, ori2augm_val=None,
                            dpool_augm_dataset=None, dpool_augm_inds=None, ori2augm=None):
    """
        Train a transformer model for an AL iteration
    :param args: arguments
    :param X_inds: indices of original training dataset used for training
    :param X_val_inds: indices of original validation dataset used during training
    :param iteration: current AL iteration
    :param val_acc_previous: accuracy of previous AL iteration
    :param uda_augm_dataset: (TensorDataset) with pairs (ori, augm) for consistency training
    :param eval_dataset: ?
    :param val_augm_dataset: (TensorDataset) with augmentations of val set
    :param test_augm_dataset:
    :param ori2augm_val: (dict) maps indices from original val set to augmented
    :param dpool_augm_dataset: (TensorDataset) with augmentations of dpool set
    :param dpool_augm_inds: (list) inds of augmented data to include in training set
    :return:
    """
    if iteration is not None:
        create_dir(args.output_dir)
        args.current_output_dir = os.path.join(args.output_dir, 'iter-{}'.format(iteration))
        args.previous_output_dir = os.path.join(args.output_dir, 'iter-{}'.format(iteration - 1))

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
    )

    logger.info("Training/evaluation parameters %s", args)
    # select after how many steps will evaluate during training
    # so that we will evaluate at least 5 times in one epoch
    minibatch = int(len(X_inds) / (args.per_gpu_train_batch_size * max(1, args.n_gpu)))
    args.logging_steps = min(int(minibatch / 5), 500)
    if args.logging_steps < 1:
        args.logging_steps = 1

    # convert to tensor dataset
    train_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)

    assert len(train_dataset) == len(X_inds)

    if eval_dataset is None:
        eval_dataset = get_glue_tensor_dataset(X_val_inds, args, args.task_name, tokenizer, evaluate=True)
    if adversarial_val_inds is not None and adversarial_val_inds != [] and val_augm_dataset is not None:
        tensor_lists = [torch.cat((eval_dataset.tensors[i], val_augm_dataset.tensors[i][adversarial_val_inds]), 0)
                        for i in range(len(eval_dataset.tensors))]
        new_eval_dataset = TensorDataset(*tensor_lists)
        eval_dataset = new_eval_dataset
    times_trained = 0
    val_acc_current = 0
    if val_acc_previous is None:
        val_acc_previous = 0.51
    val_acc_list = []
    results_list = []
    train_loss_list = []
    val_loss_list = []
    original_output_dir = args.current_output_dir

    logger.info("val acc current {}, val_acc previous {}".format(val_acc_current, val_acc_previous))

    while (val_acc_current < val_acc_previous - 0.5 and times_trained < 2) or (times_trained == 0):
        # while val_acc_current < val_acc_previous - 0.005 and times_trained < 1:
        times_trained += 1

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.current_output_dir = original_output_dir + '_trial{}'.format(times_trained)

        model.to(args.device)
        # Train
        model, train_loss, val_loss, results = train_transformer(args, train_dataset,
                                                                 eval_dataset,
                                                                 model, tokenizer,
                                                                 X_val_inds=X_val_inds)
        accuracy = results['acc']

        val_acc_current = accuracy
        if args.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass
        val_acc_list.append((times_trained, val_acc_current))
        results_list.append(results)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    best_trial = max(val_acc_list, key=lambda item: item[1])[0]
    train_loss = train_loss_list[best_trial - 1]
    results = results_list[best_trial - 1]
    best_model_ckpt = original_output_dir + '_trial{}'.format(best_trial)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_ckpt)
    model.to(args.device)
    if os.path.isdir(original_output_dir):
        shutil.rmtree(original_output_dir)
    os.rename(best_model_ckpt, original_output_dir)
    args.current_output_dir = original_output_dir

    # Results
    dataset_name = args.dataset_name
    train_results = {'model': model, 'train_loss': round(train_loss, 4), 'times_trained': times_trained}

    if iteration is None:
        path = os.path.join(EXP_DIR, '{}_{}_100%'.format(dataset_name, args.model_type))
        create_dir(path)

        print('Saving json with the results....')
        create_dir(os.path.join(path, 'det_{}'.format(args.seed)))
        with open(os.path.join(path, 'det_{}'.format(args.seed), 'results.json'), 'w') as f:
            json.dump(results, f)

    train_results.update(results)

    if train_results['acc'] > args.acc_best:
        args.acc_best_iteration = iteration
        args.acc_best = train_results['acc']
        args.best_output_dir = args.current_output_dir

    iteration_dirs = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/", recursive=True)))
    for dir in iteration_dirs:
        if dir not in [args.current_output_dir, args.best_output_dir, args.output_dir]:
            shutil.rmtree(dir)
    # train_results['val_adv_inds'] = adversarial_val_inds
    return train_results


def test_transformer_model(args, X_inds=None, model=None, ckpt=None, augm_dataset=None, dataset=None,
                           return_mean_embs=False, return_mean_output=False, return_cls=False):
    """
    Test transformer model on Dpool during an AL iteration
    :param args: arguments
    :param X_inds: indices of original *train* set
    :param model: model used for evaluation
    :param ckpt: path to model checkpoint
    :return:
    """
    if dataset is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
            use_fast=args.use_fast_tokenizer,
        )
        if augm_dataset is None:
            # it is not the eval dataset it's the Dpool
            dpool_dataset = get_glue_tensor_dataset(X_inds, args, args.task_name, tokenizer, train=True)
        else:
            # need to filter based on X_inds
            dpool_dataset = augm_dataset
    else:
        # dataset to test model on
        dpool_dataset = dataset
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.to(args.device)
    print('MC samples N={}'.format(args.mc_samples))
    result, logits = my_evaluate(dpool_dataset, args, model, al_test=True, mc_samples=args.mc_samples,
                                 return_mean_embs=return_mean_embs,
                                 return_mean_output=return_mean_output,
                                 return_cls=return_cls)
    eval_loss = result['loss']
    return eval_loss, logits, result
