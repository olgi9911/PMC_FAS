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
from utils import *
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import ConcatDataset

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

log_step = 100
model_save_epoch = 1

device_id = 'cuda:0'
root='/var/mplab_share_data'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')
results_path = '/shared/yitinglin/PMC/' + results_filename
os.system("rm -r "+results_path)
mkdir(results_path)
mkdir('/home/s113062513/PMC/logger/')
file_handler = logging.FileHandler(filename='/home/s113062513/PMC/logger/'+ results_filename +'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Train on {source}")


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

iternum = max(len(source_loader), len(target_loader))
logging.info(f"iternum={iternum}")
source_loader = get_inf_iterator(source_loader)
target_loader = get_inf_iterator(target_loader)


dann_rgb.train()
dann_ir.train()
dann_depth.train()

for epoch in range(args.epochs):

    start_steps = epoch * iternum # len(source_loader)
    total_steps = args.epochs * iternum # len(target_loader)

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
        # ============ Source domain ============ #

        # ============ Target domain ============ #
        _, domain_pred_rgb = dann_rgb(normalize_imagenet(rgb_img_target.to(device_id)), alpha, source=False)
        _, domain_pred_ir = dann_ir(normalize_imagenet(ir_img_target.to(device_id)), alpha, source=False)
        _, domain_pred_depth = dann_depth(normalize_imagenet(depth_img_target.to(device_id)), alpha, source=False)
        
        domain_loss_rgb += classifier_criterion(domain_pred_rgb, torch.ones(domain_pred_rgb.shape[0]).long().to(device_id))
        domain_loss_ir += classifier_criterion(domain_pred_ir, torch.ones(domain_pred_ir.shape[0]).long().to(device_id))
        domain_loss_depth += classifier_criterion(domain_pred_depth, torch.ones(domain_pred_depth.shape[0]).long().to(device_id))
        # ============ Target domain ============ #

        total_loss_rgb = class_loss_rgb + trade_off * domain_loss_rgb
        total_loss_ir = class_loss_ir + trade_off * domain_loss_ir
        total_loss_depth = class_loss_depth + trade_off * domain_loss_depth

        optimizer_rgb = optimizer_scheduler(optimizer_rgb, p=p)
        optimizer_ir = optimizer_scheduler(optimizer_ir, p=p)
        optimizer_depth = optimizer_scheduler(optimizer_depth, p=p)
        
        optimizer_rgb.zero_grad()
        total_loss_rgb.backward()
        optimizer_rgb.step()
        
        optimizer_ir.zero_grad()
        total_loss_ir.backward()
        optimizer_ir.step()
        
        optimizer_depth.zero_grad()
        total_loss_depth.backward()
        optimizer_depth.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  rgb_loss: %.4f  ir_loss: %.4f  depth_loss: %.4f  total_loss: %.4f'
                         % (epoch + 1, step + 1, total_loss_rgb.item(), total_loss_ir.item(), total_loss_depth.item(), total_loss_rgb.item() + total_loss_ir.item() + total_loss_depth.item()))
            
    if (epoch + 1) % model_save_epoch == 0:
        torch.save(dann_rgb.state_dict(), results_path + '/dann_rgb_epoch{}.pth'.format(epoch + 1))
        torch.save(dann_ir.state_dict(), results_path + '/dann_ir_epoch{}.pth'.format(epoch + 1))
        torch.save(dann_depth.state_dict(), results_path + '/dann_depth_epoch{}.pth'.format(epoch + 1))        
        