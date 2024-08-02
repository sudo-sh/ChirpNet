import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import glob
from tqdm import tqdm
import cv2
import sys

from h5_dataloader import *
from dataloader_utils import *


class CustomDataset(Dataset):
    def __init__(self, h5_dir, label_dir, image_flag = False, radar_flag = False, saved_loader = False, device="cuda:1", bits=None, noise_level = None ):
        print("Loading data")

        if(saved_loader is False):
            print("Creating saved dataloader folder in saved_dl")
            os.system("mkdir -p saved_dl/")

        if(saved_loader):
            self.data = np.load("saved_dl/radar_data.npy")
            self.segmap_gray = np.load("saved_dl/segmap_gray.npy")
            self.image_flag = image_flag
            self.radar_flag = radar_flag
        else:
            self.h5_dir = h5_dir
            self.label_dir = label_dir
            self.h5_files = []
            ##############################################Hardcoded#######################################################
            for h5_files in glob.glob(os.path.join(self.h5_dir, '*.h5')):
                # print(h5_files)
                self.h5_files.append(h5_files) 

            '''
            Now for each h5 file and each of the label file we will get the radar adc data
            and the label annotations.
            Currently the data is only supported for the 30m configuration with the data shape 
            as (N, 64, 8, 192) where 8 is the number of antennas 192 is the number of adc samples and 
            64 is the number of chirps
            '''
            if(radar_flag):
                # (batch, 8, 128, 192). Phase and magnitude are concatenated
                self.data = np.zeros((1, 64, 16, 192), dtype = np.float32)
            # self.labels = {}
            if(image_flag):
                self.data_rgb = np.zeros((1,3, 720,1280),dtype=np.uint8)

            self.segmap_gray = np.zeros((1,126,224), dtype= np.float32)

            for i in tqdm(range(0, len(self.h5_files))):
                if(radar_flag):
                    print("files", self.h5_files[i])
                    data_h5 = H5DatasetLoader(self.h5_files[i])
                    radar_data = torch.tensor(np.array(data_h5['radar']))
                    print("radar_data", radar_data.shape)
                    magnitude = torch.abs(radar_data)
                    phase = torch.angle(radar_data)
                    processed_data = torch.cat((magnitude, phase), dim=2)
                    # radar_data = np.transpose(radar_data,(0,2,1,3))
                    '''
                    GPU
                    '''
                    self.data = np.concatenate((self.data,processed_data))
                    print("data_shape", self.data.shape)
                    del radar_data

                #Load numpy file with the seg map
                segmap_file = os.path.join(self.label_dir, self.h5_files[i].split("/")[-1].replace(".bag.export.h5","_labels_nms.npy"))
                segmap_np = np.load(segmap_file)
                '''
                    Change the seg_map 0 for no object and 1 for object. We will do binary segmentation
                '''
                segmap_np = 1 - (1 - segmap_np // 255)
                # self.segmap_gray = segmap_np
                self.segmap_gray = np.concatenate((self.segmap_gray, segmap_np))
                ''' 
                labels structure is 
                {FRAME_ID: {'bounding_box':[[xmin,ymin,xmax,ymax]..], 'label':(tensor with single element corresponding to each box)}}

                '''
                # self.labels = merge_label_dicts(dict1=self.labels, dict2=labels)
                if(image_flag):
                    image_np = data_h5['rgb']
                    image_np = np.transpose(image_np, (0, 3, 1, 2))          
                    self.data_rgb = np.concatenate((self.data_rgb, image_np))         

            if(radar_flag):
                # print("Flag is on")
                self.data = self.data[1:]
                mean = self.data.mean()
                std = self.data.std()
                normalized_tensor = (self.data -mean)/std
                self.data = normalized_tensor
                # print norm of the data                
                np.save("saved_dl/radar_data.npy", self.data)

            if(image_flag):
                self.data_rgb = self.data_rgb[1:]

            self.segmap_gray = self.segmap_gray[1:]
            np.save("saved_dl/segmap_gray.npy", self.segmap_gray)
            self.image_flag = image_flag
            self.radar_flag = radar_flag

        
        
        if bits is not None:
            print("Norm of the data is {}".format(np.linalg.norm(self.data)))
            print("Simulating ADC with {} bits".format(bits))
            self.data = simulate_adc(self.data, bits)
            print("Norm of the data after ADC {}".format(np.linalg.norm(self.data)))

        if noise_level is not None:
            print("Norm of the data is {}".format(np.linalg.norm(self.data)))
            print("Adding noise with standard deviation {}".format(noise_level))
            self.data = self.data + np.random.normal(0, noise_level, self.data.shape)
            print("Norm of the data after adding noise {}".format(np.linalg.norm(self.data)))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get radar data for the given index
        if(self.radar_flag):
            radar_data = self.data[idx]
        if(self.image_flag):
            img = self.data_rgb[idx]
        
        segmap = self.segmap_gray[idx]

        if(self.radar_flag):
            return radar_data, segmap
        if(self.image_flag):
            return img, segmap

def main():
    h5_dir = '../data/30m_h5/filtered_less/'
    label_dir = '../data/label_dict/'

    dataset = CustomDataset(h5_dir, label_dir, radar_flag=True)
    # exit()

    for idx in tqdm(range(len(dataset))):
        radar_data, seg_map = dataset[idx]
        # img_data, seg_map = dataset[idx]
        print(radar_data.shape)
        # print(radar_data[0,0,:])
        # print(radar_data[0,1,:])
        # print(img_data.shape)
        print(seg_map.shape)
        # print(seg_map)
        # print(label_data)
        exit()

if __name__ == "__main__":
    main()