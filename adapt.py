import torch
from models.dann import DANN
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import random
import logging
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as T
from dataloader import *
import warnings
warnings.filterwarnings("ignore")
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()
import random 
import argparse
import gc
from utils import *
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0015)
parser.add_argument("--trade_off", type=float, default=0.3)
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
learning_rate = args.lr
batch_size = args.batch_size
trade_off = args.trade_off
E = args.epochs

log_step = 100
model_save_epoch = 1

device_id = 'cuda:0'
root='/var/mplab_share_data'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')
results_path = '/shared/yitinglin/PMC/' + results_filename
mkdir(results_path + '_PMC')
file_handler = logging.FileHandler(filename='/home/s113062513/PMC/logger/'+ results_filename +'_adapt.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Train on {source}")
logging.info(f"Adapt to {target}")


dann_rgb = DANN(num_classes=2, num_domains=2).to(device_id)
dann_ir = DANN(num_classes=2, num_domains=2).to(device_id)
dann_depth = DANN(num_classes=2, num_domains=2).to(device_id)

classifier_criterion = nn.CrossEntropyLoss().cuda()

optimizer_rgb = optim.SGD([
    {'params': dann_rgb.feature.parameters(), 'lr': learning_rate},
    {'params': dann_rgb.class_classifier.parameters(), 'lr': learning_rate*10},
    {'params': dann_rgb.domain_classifier.parameters(), 'lr': learning_rate*10}
], momentum=0.9)

optimizer_ir = optim.SGD([
    {'params': dann_ir.feature.parameters(), 'lr': learning_rate},
    {'params': dann_ir.class_classifier.parameters(), 'lr': learning_rate*10},
    {'params': dann_ir.domain_classifier.parameters(), 'lr': learning_rate*10}
], momentum=0.9)

optimizer_depth = optim.SGD([
    {'params': dann_depth.feature.parameters(), 'lr': learning_rate},
    {'params': dann_depth.class_classifier.parameters(), 'lr': learning_rate*10},
    {'params': dann_depth.domain_classifier.parameters(), 'lr': learning_rate*10}
], momentum=0.9)


for protocol in source.split('/'):
    source_dataset = FAS_Dataset(root=root, protocol=[protocol], train=True)
    if protocol == source.split('/')[0]:
        combined_source_dataset = source_dataset
    else:
        combined_source_dataset = ConcatDataset([combined_source_dataset, source_dataset])
source_loader = Dataloader(combined_source_dataset, batch_size=batch_size, shuffle=True)

for protocol in target.split('/'):
    target_dataset = FAS_Dataset(root=root, protocol=[protocol], train=True)
    if protocol == target.split('/')[0]:
        combined_target_dataset = target_dataset
    else:
        combined_target_dataset = ConcatDataset([combined_target_dataset, target_dataset])
target_loader = Dataloader(combined_target_dataset, batch_size=batch_size, shuffle=True)

num_source_samples = len(source_loader.dataset)
num_target_samples = len(target_loader.dataset)
len_target_loader = len(target_loader)
iternum = max(len(source_loader), len(target_loader))
logging.info(f"iternum={iternum}")
source_loader = get_inf_iterator(source_loader)
target_loader = get_inf_iterator(target_loader)

length = int(len(glob.glob(results_path + '/*.pth')) / 3) # 3 modalities

dann_rgb.train()
dann_ir.train()
dann_depth.train()

dann_rgb.load_state_dict(torch.load(results_path + '/dann_rgb_epoch' + str(length) + '.pth'))
dann_ir.load_state_dict(torch.load(results_path + '/dann_ir_epoch' + str(length) + '.pth'))
dann_depth.load_state_dict(torch.load(results_path + '/dann_depth_epoch' + str(length) + '.pth'))

r_rgb = 1 / E
r_ir = 1 / E
r_depth = 1 / E
r_fused = 1 / E

accuracy_rgb = [] # Accuracy for each epoch
accuracy_ir = []
accuracy_depth = []
accuracy_fused = []
accuracy_first_i_rgb = [] # Average accuracy for first i epochs
accuracy_first_i_ir = []
accuracy_first_i_depth = []
accuracy_first_i_fused = []

for epoch in range(args.epochs):

    start_steps = epoch * iternum # len(source_loader)
    total_steps = args.epochs * iternum # len(target_loader)

    correct_rgb = 0
    correct_ir = 0
    correct_depth = 0
    correct_fused = 0

    top_r_samples_rgb = []
    top_r_samples_ir = []
    top_r_samples_depth = []
    top_r_samples_fused = []

    for step in range(iternum):
        p = float(step + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # ============ One batch extraction ============ #
        rgb_img_source, depth_img_source, ir_img_source, labels = next(source_loader)
        rgb_img_target, depth_img_target, ir_img_target, _ = next(target_loader)

        # ============ Source domain ============ #
        class_pred_rgb, domain_pred_rgb = dann_rgb(normalize_imagenet(rgb_img_source.to(device_id)), alpha)
        class_pred_ir, domain_pred_ir = dann_ir(normalize_imagenet(ir_img_source.to(device_id)), alpha)
        class_pred_depth, domain_pred_depth = dann_depth(normalize_imagenet(depth_img_source.to(device_id)), alpha)

        class_loss_rgb = classifier_criterion(class_pred_rgb, labels.to(device_id))
        class_loss_ir = classifier_criterion(class_pred_ir, labels.to(device_id))
        class_loss_depth = classifier_criterion(class_pred_depth, labels.to(device_id))

        domain_loss_rgb = classifier_criterion(domain_pred_rgb, torch.zeros(domain_pred_rgb.shape[0]).long().to(device_id))
        domain_loss_ir = classifier_criterion(domain_pred_ir, torch.zeros(domain_pred_ir.shape[0]).long().to(device_id))
        domain_loss_depth = classifier_criterion(domain_pred_depth, torch.zeros(domain_pred_depth.shape[0]).long().to(device_id))
        
        # Calculate accuracy
        correct_rgb += (class_pred_rgb.argmax(dim=1) == labels.to(device_id)).sum().item()
        correct_ir += (class_pred_ir.argmax(dim=1) == labels.to(device_id)).sum().item()
        correct_depth += (class_pred_depth.argmax(dim=1) == labels.to(device_id)).sum().item()
        correct_fused += ((class_pred_rgb + class_pred_ir + class_pred_depth).argmax(dim=1) == labels.to(device_id)).sum().item()
        # ============ Source domain ============ #

        # ============ Target domain ============ #
        class_pred_rgb, domain_pred_rgb = dann_rgb(normalize_imagenet(rgb_img_target.to(device_id)), alpha)
        class_pred_ir, domain_pred_ir = dann_ir(normalize_imagenet(ir_img_target.to(device_id)), alpha)
        class_pred_depth, domain_pred_depth = dann_depth(normalize_imagenet(depth_img_target.to(device_id)), alpha)
        
        domain_loss_rgb += classifier_criterion(domain_pred_rgb, torch.ones(domain_pred_rgb.shape[0]).long().to(device_id))
        domain_loss_ir += classifier_criterion(domain_pred_ir, torch.ones(domain_pred_ir.shape[0]).long().to(device_id))
        domain_loss_depth += classifier_criterion(domain_pred_depth, torch.ones(domain_pred_depth.shape[0]).long().to(device_id))
        # ============ Target domain ============ #

        # ============ Select top-r_m% pseudo-labeled target samples with high classification confidence scores ============ #
        if step < len_target_loader:
            softmax_values_rgb, pseudo_labels_rgb = F.softmax(class_pred_rgb, dim=1).max(dim=1)
            softmax_values_ir, pseudo_labels_ir = F.softmax(class_pred_ir, dim=1).max(dim=1)
            softmax_values_depth, pseudo_labels_depth = F.softmax(class_pred_depth, dim=1).max(dim=1)
            softmax_values_fused, pseudo_labels_fused = F.softmax(class_pred_rgb + class_pred_ir + class_pred_depth, dim=1).max(dim=1)

            def select_top_r_samples(top_r_samples, softmax_values, img_target, pseudo_labels, r, num_target_samples):
                samples = [(softmax_values[i].item(), img_target[i], pseudo_labels[i].item()) for i in range(len(softmax_values))]
                top_r_samples.extend(samples)
                return sorted(top_r_samples, key=lambda x: x[0], reverse=True)[:int(r * num_target_samples)]

            top_r_samples_rgb = select_top_r_samples(top_r_samples_rgb, softmax_values_rgb, rgb_img_target, pseudo_labels_rgb, r_rgb, num_target_samples)
            top_r_samples_ir = select_top_r_samples(top_r_samples_ir, softmax_values_ir, ir_img_target, pseudo_labels_ir, r_ir, num_target_samples)
            top_r_samples_depth = select_top_r_samples(top_r_samples_depth, softmax_values_depth, depth_img_target, pseudo_labels_depth, r_depth, num_target_samples)
            samples = [(softmax_values_fused[i].item(), rgb_img_target[i], ir_img_target[i], depth_img_target[i], pseudo_labels_fused[i].item()) for i in range(len(softmax_values_fused))]
            top_r_samples_fused.extend(samples)
            top_r_samples_fused = sorted(top_r_samples_fused, key=lambda x: x[0], reverse=True)[:int(r_fused * num_target_samples)]
        # ============ Select top r^M% pseudo-labeled target samples with high classification confidence scores ============ #


        total_loss_rgb = class_loss_rgb + trade_off * domain_loss_rgb
        total_loss_ir = class_loss_ir + trade_off * domain_loss_ir
        total_loss_depth = class_loss_depth + trade_off * domain_loss_depth

        optimizer_rgb = optimizer_scheduler(optimizer_rgb, p=p)
        optimizer_ir = optimizer_scheduler(optimizer_ir, p=p)
        optimizer_depth = optimizer_scheduler(optimizer_depth, p=p)
        
        for optimizer, loss in zip([optimizer_rgb, optimizer_ir, optimizer_depth], 
                                    [total_loss_rgb, total_loss_ir, total_loss_depth]):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  rgb_loss: %.4f  ir_loss: %.4f  depth_loss: %.4f  total_loss: %.4f'
                         % (epoch + 1, step + 1, total_loss_rgb.item(), total_loss_ir.item(), total_loss_depth.item(), total_loss_rgb.item() + total_loss_ir.item() + total_loss_depth.item()))

    # Calculate average accuracy for the current epoch
    accuracy_rgb.append(correct_rgb / num_source_samples)
    accuracy_ir.append(correct_ir / num_source_samples)
    accuracy_depth.append(correct_depth / num_source_samples)
    accuracy_fused.append(correct_fused / num_source_samples)
    accuracy_first_i_rgb.append(sum(accuracy_rgb) / len(accuracy_rgb))
    accuracy_first_i_ir.append(sum(accuracy_ir) / len(accuracy_ir))
    accuracy_first_i_depth.append(sum(accuracy_depth) / len(accuracy_depth))
    accuracy_first_i_fused.append(sum(accuracy_fused) / len(accuracy_fused))

    eta_rgb = 0
    eta_ir = 0
    eta_depth = 0
    eta_fused = 0

    # Adjust r_m based on accuracy
    if epoch > 0:
        for i in range(1, epoch + 1):
            eta_rgb += -1 if accuracy_rgb[i] < accuracy_first_i_rgb[i] and accuracy_rgb[i - 1] < accuracy_first_i_rgb[i - 1] else 1
            eta_ir += -1 if accuracy_ir[i] < accuracy_first_i_ir[i] and accuracy_ir[i - 1] < accuracy_first_i_ir[i - 1] else 1
            eta_depth += -1 if accuracy_depth[i] < accuracy_first_i_depth[i] and accuracy_depth[i - 1] < accuracy_first_i_depth[i - 1] else 1
            eta_fused += -1 if accuracy_fused[i] < accuracy_first_i_fused[i] and accuracy_fused[i - 1] < accuracy_first_i_fused[i - 1] else 1
            
        r_rgb = np.clip(eta_rgb / E, 0, 1)
        r_ir = np.clip(eta_ir / E, 0, 1)
        r_depth = np.clip(eta_depth / E, 0, 1)
        r_fused = np.clip(eta_fused / E, 0, 1)

    # At the end of each epoch, create new DataLoader objects from the selected samples and their pseudo labels
    # Modality-Specific Sample Selection (MSS)
    def MSS():
        if len(top_r_samples_rgb) > 0:
            confidence_scores, rgb_samples, rgb_labels = zip(*[(softmax, sample, label) for softmax, sample, label in top_r_samples_rgb])
            rgb_loader = DataLoader(TensorDataset(torch.stack(rgb_samples), torch.tensor(rgb_labels), torch.tensor(confidence_scores)), batch_size=batch_size, shuffle=True)
            for rgb_img, pseudo_labels, confidence in rgb_loader:
                class_pred_rgb, _ = dann_rgb(normalize_imagenet(rgb_img.to(device_id)), alpha) # alpha does not affect the image classifier
                class_loss_rgb = (confidence.to(device_id) * classifier_criterion(class_pred_rgb, pseudo_labels.to(device_id))).mean()

                optimizer_rgb.zero_grad()
                class_loss_rgb.backward()
                optimizer_rgb.step()

        if len(top_r_samples_ir) > 0:
            confidence_scores, ir_samples, ir_labels = zip(*[(softmax, sample, label) for softmax, sample, label in top_r_samples_ir])
            ir_loader = DataLoader(TensorDataset(torch.stack(ir_samples), torch.tensor(ir_labels), torch.tensor(confidence_scores)), batch_size=batch_size, shuffle=True)
            for ir_img, pseudo_labels, confidence in ir_loader:
                class_pred_ir, _ = dann_ir(normalize_imagenet(ir_img.to(device_id)), alpha)
                class_loss_ir = (confidence.to(device_id) * classifier_criterion(class_pred_ir, pseudo_labels.to(device_id))).mean()

                optimizer_ir.zero_grad()
                class_loss_ir.backward()
                optimizer_ir.step()

        if len(top_r_samples_depth) > 0:
            confidence_scores, depth_samples, depth_labels = zip(*[(softmax, sample, label) for softmax, sample, label in top_r_samples_depth])
            depth_loader = DataLoader(TensorDataset(torch.stack(depth_samples), torch.tensor(depth_labels), torch.tensor(confidence_scores)), batch_size=batch_size, shuffle=True)
            for depth_img, pseudo_labels, confidence in depth_loader:
                class_pred_depth, _ = dann_depth(normalize_imagenet(depth_img.to(device_id)), alpha)
                class_loss_depth = (confidence.to(device_id) * classifier_criterion(class_pred_depth, pseudo_labels.to(device_id))).mean()

                optimizer_depth.zero_grad()
                class_loss_depth.backward()
                optimizer_depth.step()

    # Modality-Integrated Sample Selection (MIS)
    def MIS():
        if len(top_r_samples_fused) > 0:
            confidence_scores, rgb_samples, ir_samples, depth_samples, fused_labels = zip(*[(softmax, rgb_sample, ir_sample, depth_sample, label) for softmax, rgb_sample, ir_sample, depth_sample, label in top_r_samples_fused])
            fused_loader = DataLoader(TensorDataset(torch.stack(rgb_samples), torch.stack(ir_samples), torch.stack(depth_samples), torch.tensor(fused_labels), torch.tensor(confidence_scores)), batch_size=batch_size, shuffle=True)
            for rgb_img, ir_img, depth_img, pseudo_label, confidence in fused_loader:
                class_pred_rgb, _ = dann_rgb(normalize_imagenet(rgb_img.to(device_id)), alpha)
                class_pred_ir, _ = dann_ir(normalize_imagenet(ir_img.to(device_id)), alpha)
                class_pred_depth, _ = dann_depth(normalize_imagenet(depth_img.to(device_id)), alpha)
                class_loss_rgb = (confidence.to(device_id) * classifier_criterion(class_pred_rgb, pseudo_label.to(device_id))).mean()
                class_loss_ir = (confidence.to(device_id) * classifier_criterion(class_pred_ir, pseudo_label.to(device_id))).mean()
                class_loss_depth = (confidence.to(device_id) * classifier_criterion(class_pred_depth, pseudo_label.to(device_id))).mean()

                for optimizer, loss in zip([optimizer_rgb, optimizer_ir, optimizer_depth], 
                                            [class_loss_rgb, class_loss_ir, class_loss_depth]):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    MSS()
    del top_r_samples_rgb, top_r_samples_ir, top_r_samples_depth
    gc.collect()
    MIS()

    if (epoch + 1) % model_save_epoch == 0:
        torch.save(dann_rgb.state_dict(), results_path + '_PMC/dann_rgb_epoch{}.pth'.format(epoch + 1))
        torch.save(dann_ir.state_dict(), results_path + '_PMC/dann_ir_epoch{}.pth'.format(epoch + 1))
        torch.save(dann_depth.state_dict(), results_path + '_PMC/dann_depth_epoch{}.pth'.format(epoch + 1))        
        