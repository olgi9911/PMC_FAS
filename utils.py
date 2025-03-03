import torch
import numpy as np
import os
import heapq
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def select_top_r_samples(top_r_samples, softmax_values, pseudo_labels, r, num_target_samples, start_idx):
    samples = [(softmax_values[i].item(), start_idx + i, pseudo_labels[i].item()) for i in range(len(softmax_values))]
    for item in samples:
        if len(top_r_samples) < r * num_target_samples:
            heapq.heappush(top_r_samples, item)
        else:
            heapq.heappushpop(top_r_samples, item)
    return top_r_samples

def create_subset_loader(top_r_samples, dataset, shuffled_indices, batch_size):
        confidence_scores, indices, pseudo_labels = zip(*[(softmax, index, label) for softmax, index, label in top_r_samples])
        confidence_scores = torch.tensor(confidence_scores)
        pseudo_labels = torch.tensor(pseudo_labels)
        target_subset = Subset(dataset, [shuffled_indices[i] for i in indices])
        return DataLoader(target_subset, batch_size=batch_size, shuffle=False), confidence_scores, pseudo_labels


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_data_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def normalize_imagenet(img):
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # img = TF.to_tensor(img)
    img = TF.normalize(img, mean=mean, std=std)
    return img

def calculate_tpr_at_fpr(labels, predictions, target_fpr=0.001):
    sorted_indices = np.argsort(predictions)[::-1].astype(int)
    
    sorted_labels = np.array(labels)[sorted_indices]

    TP = np.cumsum(sorted_labels)
    FP = np.cumsum(1 - sorted_labels)
    FN = np.sum(sorted_labels) - TP
    TN = len(sorted_labels) - np.sum(sorted_labels) - FP

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    target_index = np.where(FPR <= target_fpr)[0]
    if len(target_index) == 0:
        return None  # No FPR value is as low as the target
    tpr_at_target_fpr = TPR[target_index[-1]]

    return tpr_at_target_fpr

def calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001):
    interpolated_tpr = np.interp(fpr_threshold, fpr, tpr)
    return interpolated_tpr

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

    
''' Stolen from https://github.com/NaJaeMin92/pytorch-DANN'''
def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None