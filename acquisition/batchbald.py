"""
Code from https://github.com/BlackHC/BatchBALD
"""
import math
import os
import sys

import torch
import numpy as np
# from blackhc.progress_bar import with_progress_bar

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import acquisition.BatchBALD.src.joint_entropy.exact as joint_entropy_exact
import acquisition.BatchBALD.src.joint_entropy.sampling as joint_entropy_sampling
from acquisition.BatchBALD.src import torch_utils



# from modules.acquisition.BatchBALD.src import torch_utils
# from blackhc.progress_bar import with_progress_bar
# import modules.acquisition.BatchBALD.src.joint_entropy.exact as joint_entropy_exact
# import modules.acquisition.BatchBALD.src.joint_entropy.sampling as joint_entropy_sampling
# from modules.acquisition.BatchBALD.src.acquisition_batch import AcquisitionBatch

compute_multi_bald_bag_multi_bald_batch_size = None

def with_progress_bar(
    iterable, length=None, length_unit=None, unit_scale=None, tqdm_args=None
):
    return ProgressBarIterable(
        iterable,
        length=length,
        length_unit=length_unit,
        unit_scale=unit_scale,
        tqdm_args=tqdm_args,
    )

def batch_exact_joint_entropy_logits(logits_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, logits_b_K_C in with_progress_bar(
            torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(logits_b_K_C.to(device).exp(), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, device, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in with_progress_bar(
            torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size), unit_scale=chunk_size
    ):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(probs_b_K_C.to(device), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b


def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])


def mutual_information(logits_B_K_C):
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def BB_acquisition(logits_B_K_C, device, b):
    """

    :param logits_B_K_C: list of tensors with logits after MC dropout.
    - B: |Dpool|
    - K: number of MC samples
    - C: number of classes
    :return:
    """

    if type(logits_B_K_C) is list:
        logits_B_K_C = torch.stack(logits_B_K_C, 1)

    bald_scores = mutual_information(logits_B_K_C)

    partial_multi_bald_B = bald_scores

    k = logits_B_K_C.size(1)
    num_classes = logits_B_K_C.size(2)

    # Now we can compute the conditional entropy
    conditional_entropies_B = joint_entropy_exact.batch_conditional_entropy_B(logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = logits_B_K_C.exp_()

    torch_utils.gc_cuda()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        if device.type == "cuda":
            # KC_memory = k*num_classes*8
            sample_MK_memory = num_samples * k * 8
            MC_memory = num_samples * num_classes * 8
            copy_buffer_memory = 256 * num_samples * num_classes * 8
            slack_memory = 2 * 2 ** 30
            multi_bald_batch_size = (
                                            torch_utils.get_cuda_available_memory() - (
                                                sample_MK_memory + copy_buffer_memory + slack_memory)
                                    ) // MC_memory

            global compute_multi_bald_bag_multi_bald_batch_size
            if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
                compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size
                print(f"New compute_multi_bald_bag_multi_bald_batch_size = {multi_bald_batch_size}")
        else:
            multi_bald_batch_size = 16

        subset_acquisition_bag = []
        global_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        # for i in range(b):
        i = 0
        while i < b:
            torch_utils.gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None].to(device),
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )

                    # torch_utils.cuda_meminfo()
                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, device, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        torch_utils.gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                        probs_B_K_C[subset_acquisition_bag].to(device), S=num_samples_per_ws
                    )

                    # torch_utils.cuda_meminfo()
                    for joint_entropies_b, probs_b_K_C in with_progress_bar(
                            torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size),
                            unit_scale=multi_bald_batch_size,
                    ):
                        joint_entropies_b.copy_(
                            joint_entropy_sampling.batch(probs_b_K_C.to(device), prev_samples_M_K),
                            non_blocking=True
                        )

                        # torch_utils.cuda_meminfo()

                    prev_samples_M_K = None
                    torch_utils.gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            print(f"Actual MultiBALD: {actual_multi_bald_B}")

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    print("Early out")
                    break

            if winner_index not in global_acquisition_bag:
                acquisition_bag_scores.append(actual_multi_bald_B)


                subset_acquisition_bag.append(winner_index)
                # We need to map the index back to the actual dataset.
                # for now we keep it the same...
                # global_acquisition_bag.append(subset_split.get_dataset_indices([winner_index]).item())
                global_acquisition_bag = subset_acquisition_bag

                print(f"Acquisition bag: {sorted(global_acquisition_bag)}")
                i+= 1
            # else:
            #     b += 1
    # return AcquisitionBatch(global_acquisition_bag, acquisition_bag_scores, None)
    # return acquisition_bag_scores.cpu().numpy(), global_acquisition_bag.cpu().numpy()
    return np.asarray(acquisition_bag_scores), np.asarray(global_acquisition_bag)


def random_acquisition_function(logits_b_K_C):
    # If we use this together with a heuristic, make it small, so the heuristic takes over after the
    # first random pick.
    return torch.rand(logits_b_K_C.shape[0], device=logits_b_K_C.device) * 0.00001


def variation_ratios(logits_b_K_C):
    # torch.max yields a tuple with (max, argmax).
    return torch.ones(logits_b_K_C.shape[0], dtype=logits_b_K_C.dtype, device=logits_b_K_C.device) - torch.exp(
        torch.max(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=1, keepdim=False)[0]
    )


def mean_stddev_acquisition_function(logits_b_K_C):
    return torch_utils.mean_stddev(logits_b_K_C)


def max_entropy_acquisition_function(logits_b_K_C):
    return entropy(torch_utils.logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=-1)


def bald_acquisition_function(logits_b_K_C):
    return torch_utils.mutual_information(logits_b_K_C)


if __name__ == '__main__':
    logits = torch.rand(100, 3, 2)
    logits_B_K_C = torch.tensor([[[-0.084, 0.0803], [-0.1033, 0.0489], [-0.1125, 0.0514]],
                                 [[-0.1134, 0.0802], [-0.1242, 0.0619], [-0.90922, 0.0952]],
                                 [[-0.0942, 0.0617], [-0.0776, 0.0421], [-0.0934, 0.06009]]])

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    max_entropy_scores = max_entropy_acquisition_function(logits_B_K_C)
    least_confidence_scores = variation_ratios(logits_B_K_C)
    mean_std_scores = mean_stddev_acquisition_function(logits_B_K_C)
    # BB_acquisition(logits_B_K_C=logits, device=device, b=10, k=3, num_classes=2)
    BB_acquisition(logits_B_K_C=logits, device=device, b=10)
