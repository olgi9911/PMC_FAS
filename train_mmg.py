import torch
from models.mmg import MMG
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
parser.add_argument("--missing", type=str, default='none')
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.00015)
parser.add_argument("--trade_off", type=float, default=0.1)
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
missing = args.missing
learning_rate = args.lr
batch_size = args.batch_size
trade_off = args.trade_off

log_step = 100
model_save_epoch = 1

device_id = 'cuda:0'
root='/var/mplab_share_data'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')
results_path = root + '/yitinglin/PMC/' + results_filename + '_MMG'
os.system("rm -r "+results_path)
mkdir(results_path)
# mkdir('/home/s113062513/PMC/logger/')
file_handler = logging.FileHandler(filename='/home/s113062513/PMC/logger/'+ results_filename +'_MMG_' + missing + '.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Train on {source}")


mmg = MMG(n_channels=3, n_classes=2).to(device_id)

'''Stolen from https://github.com/zhangweichen2006/PMC/blob/master/Object%20Recognition/RGBD%20Object%20Recogntion/rgbd10_RGBDepth.py'''
unet_list = list(map(id, mmg.input_block.parameters()))
unet_list.extend(list(map(id, mmg.down_blocks.parameters())))
unet_list.extend(list(map(id, mmg.down_bottleneck.parameters())))
# unet_list.extend(list(map(id, mmg.bridge.parameters())))
unet_list.extend(list(map(id, mmg.up_blocks.parameters())))
unet_list.extend(list(map(id, mmg.up_bottleneck.parameters())))
unet_list.extend(list(map(id, mmg.out.parameters())))
unet_parameters = filter(lambda p: id(p) in unet_list, mmg.parameters())

domain_classifier_list = list(map(id, mmg.domain_classifier.parameters()))
domain_classifier_list.extend(list(map(id, mmg.down_bottleneck.parameters())))
domain_classifier_parameters = filter(lambda p: id(p) in domain_classifier_list, mmg.parameters())

encoder_list = list(map(id, mmg.input_block.parameters()))
encoder_list.extend(list(map(id, mmg.down_blocks.parameters())))
encoder_parameters = filter(lambda p: id(p) in encoder_list, mmg.parameters())

optimizer_unet = optim.Adam(unet_parameters, lr=learning_rate*10, betas = (0.5, 0.999), weight_decay=0.0003)
optimizer_domain_classifier = optim.SGD(domain_classifier_parameters, lr=learning_rate*10, momentum=0.9, weight_decay=0.0003)
optimizer_encoder = optim.SGD(encoder_parameters, lr=learning_rate, momentum=0.9, weight_decay=0.0003)

generator_criterion = nn.L1Loss().cuda()
classifier_criterion = nn.BCEWithLogitsLoss().cuda()


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


mmg.train()

for epoch in range(args.epochs):

    start_steps = epoch * iternum # len(source_loader)
    total_steps = args.epochs * iternum # len(target_loader)

    epoch_gen_loss = 0.0

    for step in range(iternum):
        p = float(step + start_steps) / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # ============ One batch extraction ============ #
        rgb_img_source, depth_img_source, ir_img_source, labels = next(source_loader)

        # ============ Source domain ============ #
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        generated_modality = mmg(normalize_imagenet(rgb_img_source.to(device_id)), class_labels=labels.to(device_id), train_generator=True, train_domain_classifier=False)
        
        mmg.zero_grad()

        if missing == 'depth':
            generator_loss = generator_criterion(generated_modality, depth_img_source.to(device_id))
        elif missing == 'ir':
            generator_loss = generator_criterion(generated_modality, ir_img_source.to(device_id))

        generator_loss.backward()
        optimizer_unet.step()
        # ============ Source domain ============ #
        
        if epoch >= args.epochs // 2:
            # ============ Target domain ============ #
            rgb_img_target, depth_img_target, ir_img_target, _ = next(target_loader)
            rgb_combined = torch.cat([rgb_img_source, rgb_img_target], dim=0)
            domain_pred = mmg(normalize_imagenet(rgb_combined.to(device_id)), train_generator=False, train_domain_classifier=True)
            domain_labels = torch.cat([torch.zeros(rgb_img_source.shape[0]), torch.ones(rgb_img_target.shape[0])]).float().to(device_id)
            domain_loss = classifier_criterion(domain_pred, domain_labels)
            domain_loss *= trade_off
            domain_loss.backward()
            optimizer_domain_classifier.step()
            optimizer_encoder.step()
            # ============ Target domain ============ #


        if (step + 1) % log_step == 0 and epoch < args.epochs // 2:
            logging.info('[epoch %d step %d]  generator_loss: %.4f' % (epoch + 1, step + 1, generator_loss.item()))
        elif (step + 1) % log_step == 0 and epoch >= args.epochs // 2:
            logging.info('[epoch %d step %d]  generator_loss: %.4f  domain_loss: %.4f' % (epoch + 1, step + 1, generator_loss.item(), domain_loss.item() / trade_off))
            
    # if (epoch + 1) % model_save_epoch == 0:
    #     torch.save(mmg.state_dict(), results_path + '/dann_rgb_epoch{}.pth'.format(epoch + 1))

torch.save(mmg.state_dict(), results_path + '/mmg_' + missing +'.pth')