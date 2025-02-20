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
import sklearn.metrics as metrics
import glob
import argparse
from sklearn.metrics import roc_auc_score
from utils import *
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import ConcatDataset

np.random.seed(42)
torch.manual_seed(42)

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--test_dataset", type=str)
parser.add_argument("--missing", type=str, default='none')
parser.add_argument("--pmc", type=bool, default=False)
args = parser.parse_args()

source = args.train_dataset
target = args.test_dataset
missing = args.missing
pmc = args.pmc
batch_size = 32

model_save_epoch = 1

device_id = 'cuda:0'
root='/var/mplab_share_data'
results_filename = source.replace('/', '') + '_to_' + target.replace('/', '')
if pmc:
    results_filename += '_PMC'
    
results_path = '/shared/yitinglin/PMC/' + results_filename
file_handler = logging.FileHandler(filename='/home/s113062513/PMC/logger/'+ results_filename +'_test.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)
logging.info(f"Train on {source}")
logging.info(f"Test on {target}")

dann_rgb = DANN(num_classes=2, num_domains=2).to(device_id)
dann_ir = DANN(num_classes=2, num_domains=2).to(device_id)
dann_depth = DANN(num_classes=2, num_domains=2).to(device_id)

for protocol in target.split('/'):
    target_dataset = FAS_Dataset(root=root, protocol=[protocol], train=True)
    if protocol == target.split('/')[0]:
        combined_target_dataset = target_dataset
    else:
        combined_target_dataset = ConcatDataset([combined_target_dataset, target_dataset])
target_loader = Dataloader(combined_target_dataset, batch_size=batch_size, shuffle=True)
data_loader = target_loader

logging.info(f"# of testing: {len(data_loader)}")

record = [1,100,100,100,100,100]
length = int(len(glob.glob(results_path + '/*.pth')) / 3) # 3 modalities

log_list = []

dann_rgb.eval()
dann_ir.eval()
dann_depth.eval()
with torch.no_grad():
    for step in reversed(range(1, length)):
        dann_rgb.load_state_dict(torch.load(results_path + '/dann_rgb_epoch' + str(step) + '.pth'))
        dann_ir.load_state_dict(torch.load(results_path + '/dann_ir_epoch' + str(step) + '.pth'))
        dann_depth.load_state_dict(torch.load(results_path + '/dann_depth_epoch' + str(step) + '.pth'))

        step = step * model_save_epoch

        score_list = []
        Total_score_list_cs = []
        Total_score_list_all = []
        label_list = []
        TP = 0.0000001
        TN = 0.0000001
        FP = 0.0000001
        FN = 0.0000001

        for i, data in enumerate(data_loader):
            rgb_img, depth_img, ir_img, labels = data
            rgb_img = rgb_img.to(device_id)
            depth_img = depth_img.to(device_id)
            ir_img = ir_img.to(device_id)

            rgb_img = normalize_imagenet(rgb_img)
            depth_img = normalize_imagenet(depth_img)
            ir_img = normalize_imagenet(ir_img)

            pred_rgb, _ = dann_rgb(rgb_img, 0) # alpha = 0, which does not matter in testing
            pred_ir, _ = dann_ir(ir_img, 0)
            pred_depth, _ = dann_depth(depth_img, 0)
            pred = (pred_rgb + pred_ir + pred_depth) / 3

            score = F.softmax(pred, dim=1)[:, 1].cpu().numpy()
            
            for j in range(rgb_img.size(0)):
                score_list.append(score[j])
                label_list.append(labels[j])

        for i in range(0, len(label_list)):
            Total_score_list_cs.append(score_list[i]) 
            if score_list[i] == None: # if there is nan in Total_score_list_cs, print it out
                print(score_list[i])

        fpr, tpr, thresholds_cs = metrics.roc_curve(label_list, Total_score_list_cs)
        threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds_cs)

        for i in range(len(Total_score_list_cs)):
            score = Total_score_list_cs[i]
            if (score >= threshold_cs and label_list[i] == 1):
                TP += 1
            elif (score < threshold_cs and label_list[i] == 0):
                TN += 1
            elif (score >= threshold_cs and label_list[i] == 0):
                FP += 1
            elif (score < threshold_cs and label_list[i] == 1):
                FN += 1

        APCER = FP / (TN + FP)
        NPCER = FN / (FN + TP)
    
        if record[1]>((APCER + NPCER) / 2):
            record[0]=step + 1
            record[1]=((APCER + NPCER) / 2)
            record[2]=roc_auc_score(label_list, score_list)
            record[3]=APCER
            record[4]=NPCER
            record[5]=calculate_tpr_at_fpr(label_list, normalize_data(score_list))

        log_list.append([step, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4)])
        logging.info('[Epoch %d] APCER %.4f BPCER %.4f ACER %.4f  AUC %.4f tpr_fpr0001 %.4f'
                % (step + 1, np.round(APCER, 4), np.round(NPCER, 4), np.round((APCER + NPCER) / 2, 4), np.round(roc_auc_score(label_list, score_list), 4) , calculate_interpolated_tpr(fpr, tpr, fpr_threshold=0.001)))
            
log_list.sort(key=lambda x: x[3])
print(log_list)

logging.info(f"Modalities BEST Epoch {str(record[0])} ACER {str(record[1])} AUC {str(record[2])} APCER {str(record[3])} BPCER {str(record[4])} tpr_fpr0001 {str(record[5])}")
