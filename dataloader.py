from __future__ import print_function, division
import os
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler

def get_frame(path):
    
    frame =  Image.open(path)
    # face_frame = transform_face(frame)
    
    return frame


def getSubjects(configPath):
    
    f = open(configPath, "r")
    
    all_live, all_spoof = [], []
    while(True):
        line = f.readline()
        if not line:
            break
        line = line.strip()
        # print(line)
        
        ls, subj = line.split(",")
        if(ls == "+1"):
            all_live.append(subj)
            # print("live", subj)
        else:
            all_spoof.append(subj)
            # print("spoof", subj)
    
    print(f"{configPath=}")
    print(f"{len(all_live)=}, {len(all_spoof)=}")
    
    return all_live, all_spoof



class FAS_Dataset(Dataset):
    def __init__(self, root, protocol=['C','S','W','P'], train = True , adapt = False, size = 224):
        
        self.all_liver = []
        self.all_lived = []
        self.all_livei = []
        self.all_spoofr = []
        self.all_spoofd = []
        self.all_spoofi = []
        self.protocol = protocol
        print(f'protocol {protocol}')
        
        for i in protocol:

            if i == 'C':
                
                # self.all_liver = np.load(root + '/CeFA/real/rgb.npy')
                # self.all_lived = np.load(root + '/CeFA/real/depth.npy')
                # self.all_livei = np.load(root + '/CeFA/real/ir.npy')
                # self.all_spoofr= np.load(root + '/CeFA/spoof/rgb.npy')
                # self.all_spoofd= np.load(root + '/CeFA/spoof/depth.npy')
                # self.all_spoofi= np.load(root + '/CeFA/spoof/ir.npy')

                liver = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/real/profile/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/spoof/profile/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"domain-generalization-multi/CeFA/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
                
            if i == 'S':

                liver = sorted(glob.glob(os.path.join(root, f"SURF_intra/real/rgb/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"SURF_intra/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"SURF_intra/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"SURF_intra/spoof/rgb/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"SURF_intra/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"SURF_intra/spoof/ir/*.jpg")))
                    
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
                
                
            if i == 'P':

                self.all_liver = np.load(root + '/padisi/real/rgb.npy')
                self.all_lived = np.load(root + '/padisi/real/depth.npy')
                self.all_livei = np.load(root + '/padisi/real/ir.npy')
                self.all_spoofr= np.load(root + '/padisi/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/padisi/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/padisi/spoof/ir.npy')

            if i == 'W':

                self.all_liver = np.load(root + '/WMCA/real/rgb.npy')
                self.all_lived = np.load(root + '/WMCA/real/depth.npy')
                self.all_livei = np.load(root + '/WMCA/real/ir.npy')
                self.all_spoofr= np.load(root + '/WMCA/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/WMCA/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/WMCA/spoof/ir.npy')
        
        
        self.live_labels = np.ones(len(self.all_liver), dtype=np.int64)
        self.spoof_labels = np.zeros(len(self.all_spoofr), dtype=np.int64)
        self.total_labels = np.concatenate((self.live_labels, self.spoof_labels), axis=0)
        
        self.total_rgb = np.concatenate((self.all_liver, self.all_spoofr), axis=0)
        self.total_depth = np.concatenate((self.all_lived, self.all_spoofd), axis=0)
        self.total_ir = np.concatenate((self.all_livei, self.all_spoofi), axis=0)

        self.train = train
        self.size = size
        self.randomcrop = transforms.RandomResizedCrop(self.size)
        
    def transform(self, img1, img2, img3, train = True, size = 224):
    
        if train: # training
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (size,size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (size,size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (size,size))
           

            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)

            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                img3 = TF.hflip(img3)

            # Random vertical flipping
            if random.random() > 0.5:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                img3 = TF.vflip(img3)

            # Random rotation
            angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
            img1 = TF.rotate(img1,angle)
            img2 = TF.rotate(img2,angle)
            img3 = TF.rotate(img3,angle)

        else:  # testing
            img1 = TF.resize(img1, (size,size))
            img2 = TF.resize(img2, (size,size))
            img3 = TF.resize(img3, (size,size))
            
            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)
            
            # color_jitter  = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
            # img1 = color_jitter (img1)
            # img2 = color_jitter (img2)
            # img3 = color_jitter (img3)

        # Transform to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        img3 = TF.to_tensor(img3)
        
        return img1, img2, img3

    def __getitem__(self, idx):
        
        
        rgb = self.total_rgb[idx]   # get rgb
        depth = self.total_depth[idx]  # get depth
        ir = self.total_ir[idx]  # get ir
        
        labels = self.total_labels[idx]
        
        if self.protocol[0] == 'C' or self.protocol[0] == 'S':
            rgb_img = get_frame(rgb)
            depth_img = get_frame(depth)
            ir_img = get_frame(ir)
        else:
            rgb_img = Image.fromarray(rgb)
            depth_img = Image.fromarray(depth)
            ir_img = Image.fromarray(ir)
        
        
        rgb_img, depth_img, ir_img = self.transform(rgb_img, depth_img, ir_img, self.train, self.size)
        
        return rgb_img, depth_img, ir_img, labels


    def __len__(self):
        return len(self.total_rgb)

class FAS_Dataset_CAM(Dataset):
    def __init__(self, root, protocol=['C','S','W','P',], train = True , size = 224, cls="live"):
        
        self.all_liver = []
        self.all_lived = []
        self.all_livei = []
        self.all_spoofr = []
        self.all_spoofd = []
        self.all_spoofi = []
        self.protocol = protocol

        for i in protocol:

            if i == 'C':
                liver = sorted(glob.glob(os.path.join(root, f"CeFA/real/profile/*.jpg")))
                lived = sorted(glob.glob(os.path.join(root, f"CeFA/real/depth/*.jpg")))
                livei = sorted(glob.glob(os.path.join(root, f"CeFA/real/ir/*.jpg")))
                
                spoofr = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/profile/*.jpg")))
                spoofd = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/depth/*.jpg")))
                spoofi = sorted(glob.glob(os.path.join(root, f"CeFA/spoof/ir/*.jpg")))
                
                self.all_liver += liver
                self.all_lived += lived
                self.all_livei += livei
                
                self.all_spoofr += spoofr
                self.all_spoofd += spoofd
                self.all_spoofi += spoofi
                
            if i == 'S':

                self.all_liver = np.load(root + '/SURF_intra/real/rgb.npy')
                self.all_lived = np.load(root + '/SURF_intra/real/depth.npy')
                self.all_livei = np.load(root + '/SURF_intra/real/ir.npy')
                self.all_spoofr= np.load(root + '/SURF_intra/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/SURF_intra/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/SURF_intra/spoof/ir.npy')

            if i == 'P':

                self.all_liver = np.load(root + '/padisi/real/rgb.npy')
                self.all_lived = np.load(root + '/padisi/real/depth.npy')
                self.all_livei = np.load(root + '/padisi/real/ir.npy')
                self.all_spoofr= np.load(root + '/padisi/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/padisi/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/padisi/spoof/ir.npy')

            if i == 'W':

                self.all_liver = np.load(root + '/WMCA/real/rgb.npy')
                self.all_lived = np.load(root + '/WMCA/real/depth.npy')
                self.all_livei = np.load(root + '/WMCA/real/ir.npy')
                self.all_spoofr= np.load(root + '/WMCA/spoof/rgb.npy')
                self.all_spoofd= np.load(root + '/WMCA/spoof/depth.npy')
                self.all_spoofi= np.load(root + '/WMCA/spoof/ir.npy')

        if cls == "live":
            self.total_labels = np.ones(len(self.all_liver), dtype=np.int64)
            self.total_rgb = self.all_liver
            self.total_depth = self.all_lived
            self.total_ir = self.all_livei

        elif cls == "spoof":
            self.total_labels = np.zeros(len(self.all_spoofr), dtype=np.int64)
            self.total_rgb = self.all_spoofr
            self.total_depth = self.all_spoofd
            self.total_ir = self.all_spoofi

        elif cls == "all":
            self.live_labels = np.ones(len(self.all_liver), dtype=np.int64)
            self.spoof_labels = np.zeros(len(self.all_spoofr), dtype=np.int64)
            self.total_labels = np.concatenate((self.live_labels, self.spoof_labels), axis=0)
            
            self.total_rgb = np.concatenate((self.all_liver, self.all_spoofr), axis=0)
            self.total_depth = np.concatenate((self.all_lived, self.all_spoofd), axis=0)
            self.total_ir = np.concatenate((self.all_livei, self.all_spoofi), axis=0)
            
        self.train = train
        self.size = size
        self.randomcrop = transforms.RandomResizedCrop(self.size)
        
    def transform(self, img1, img2, img3, train = True, size = 224):
    
        if train: # training
            img1 = TF.center_crop(TF.resize(img1, (256,256)), (size,size))
            img2 = TF.center_crop(TF.resize(img2, (256,256)), (size,size))
            img3 = TF.center_crop(TF.resize(img3, (256,256)), (size,size))
           

            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)

        else:  # testing
            img1 = TF.resize(img1, (size,size))
            img2 = TF.resize(img2, (size,size))
            img3 = TF.resize(img3, (size,size))
            
            img2 = TF.rgb_to_grayscale(img2,num_output_channels=3)
            img3 = TF.rgb_to_grayscale(img3,num_output_channels=3)
            
        # Transform to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        img3 = TF.to_tensor(img3)
        
        return img1, img2, img3

    def __getitem__(self, idx):
        
        
        rgb = self.total_rgb[idx]   # get rgb
        depth = self.total_depth[idx]  # get depth
        ir = self.total_ir[idx]  # get ir
        
        labels = self.total_labels[idx]
        
        if self.protocol[0] == 'C':  # or self.protocol[0] == 'S':
            rgb_img = get_frame(rgb)
            depth_img = get_frame(depth)
            ir_img = get_frame(ir)
        else:
            rgb_img = Image.fromarray(rgb)
            depth_img = Image.fromarray(depth)
            ir_img = Image.fromarray(ir)
        
        
        rgb_img, depth_img, ir_img = self.transform(rgb_img, depth_img, ir_img, self.train, self.size)
        
        return rgb_img, depth_img, ir_img, labels


    def __len__(self):
        return len(self.total_rgb)

def get_loader(root, protocol, batch_size=10, shuffle=True, train = True, adapt = False, size = 224, CAM=False, cls="live"):
    
    if CAM:
        _dataset = FAS_Dataset_CAM(root=root,
                               protocol=protocol,
                               train = train,
                               size = size,
                               cls = cls)
    else:
        _dataset = FAS_Dataset(root=root,
                               protocol=protocol,
                               train = train,
                               size = size,
                               adapt = adapt)
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for rgb_img, depth_img, ir_img, labels in data_loader:
            yield (rgb_img, depth_img, ir_img, labels)

def create_dataloader_with_sampler(dataset, selected_indices, batch_size):
    """
    Create a DataLoader with SubsetRandomSampler.
    """
    sampler = SubsetRandomSampler(selected_indices)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

import cv2
if __name__ == "__main__":
    
    train_loader = get_loader(root = '/shared', protocol=['C'], batch_size=1800, shuffle=True)

    count = 0
    total = 0
    for i, (rgb_img, depth_img, ir_img, labels) in enumerate(train_loader):
        print(rgb_img.shape)
        print(depth_img.shape)
        print(ir_img.shape)
        total += rgb_img.shape[0]
        count += torch.sum (labels)
        
    # print number of 1labels is tensor
    print('number of 1 labels: ' + str(count))
    print(total)