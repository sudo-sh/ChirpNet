
#Import functions from other files
from models.chirpnet import *
from chirpnet_dataloader import *
from utils_train import *

from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import cv2
import os
import json
from collections import defaultdict
import numpy as np
from torchvision.ops import box_iou
import logging
import datetime
import random
import argparse

seed = 42
# Set the seed for generating random numbers
torch.manual_seed(seed)
torch.cuda.empty_cache()
# If you are using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(0)  # for multi-GPU.



def evaluate_model_advanced(model, data_loader, device, num_classes, batch_size = 8):
    model.eval()  # Set the model to evaluation mode
    mse_errors = []
    dice_coef_all = 0
    chamfer_dist_all = 0
    with torch.no_grad():  # No need to track gradients
        r_ = random.randint(0, len(data_loader)-1)
        for i, (radar_data, seg_map) in enumerate(data_loader):
        
            radar_data = radar_data.to(device).to(torch.float32)
            seg_map = seg_map.to(device).to(torch.float32)
            pred_seg_map = model(radar_data)
            
            mse_errors.append(calculate_pixelwise_mse(pred_segmap=pred_seg_map, gt_segmap=seg_map))
            if(i==r_):
                os.system("mkdir -p debug_picture")
                visualize_and_save_radar(y_true_batch=seg_map, y_pred_batch=pred_seg_map, n = 8, save_path="debug_picture/radar_bin_seg.jpg")
            
            copy_pred_seg_map = torch.sigmoid(pred_seg_map)
            copy_pred_seg_map = torch.round(copy_pred_seg_map)
            copy_pred_seg_map = copy_pred_seg_map.cpu().numpy().astype(np.uint8)
            copy_gt_segmap = seg_map.cpu().numpy().astype(np.uint8)


            dice_coef_all += dice_coefficient_batch(copy_pred_seg_map, copy_gt_segmap)

            chamfer_dist_all += chamfer_distance_batch(copy_gt_segmap[:]*255, copy_pred_seg_map[:][0]*255)

    mse_error = torch.mean(torch.tensor(mse_errors))
    dice_coef = dice_coef_all/(batch_size*len(data_loader))
    chamfer_dist = chamfer_dist_all/(batch_size*len(data_loader))
    
    return mse_error, dice_coef, chamfer_dist

def create_dataloaders(h5_dir, label_dir, batch_size=1, train_split=0.8, num_workers=0, device="cuda:0", bits=None, noise_level=None):
    # Initialize the dataset
    #Turn Saved_loader to False if using new dataset
    dataset = CustomDataset(h5_dir, label_dir, radar_flag=True, device=device, saved_loader=True, bits=bits, noise_level=noise_level)

    # Calculate train and test sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    # Generate indices for train and test sets
    indices = torch.arange(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:total_size]

    # Create train and test datasets using Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader





def load_params(file_path):
    """
    Load parameters from a JSON file.

    :param file_path: Path to the JSON file containing parameters.
    :return: Dictionary containing parameters.
    """
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params


def train(args):
    ''' 
        Load the json parameter file
    '''
    file_path = args.params_filename
    params = load_params(file_path)
    device = params["device"]


    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the date and time in a suitable format for a filename
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Set up the log file name with the current date and time
    os.system("mkdir -p "+ params["log_path"])
    log_filename = params["log_path"] + "/app_"+formatted_time+".log"
    if not os.path.exists(args.save_checkpoint_path):
        os.mkdir(args.save_checkpoint_path)


    checkpoint_model_name_folder = args.save_checkpoint_path + "model_RADFE_" + str(args.linear_dims) + "_" +str(args.hidden_dim)+"_" +str(args.conv)+"/"
    
    print("Checkpoint_folder_path", checkpoint_model_name_folder)
    if not os.path.exists(checkpoint_model_name_folder):
        os.mkdir(checkpoint_model_name_folder)

    
    # Configure logging
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    h5_dir = 'data/'
    label_dir = 'data/label_dict/'
    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths
    logging.info("[INFO] loading dataset...")

    train_loader, test_loader = create_dataloaders(h5_dir, label_dir, batch_size=params['batch_size'],\
                         train_split=0.7, num_workers=0, device = params['device'],  bits= args.bits, noise_level= args.noise_level)

    for arg, value in vars(args).items():
        logging.info(f'Argument {arg}: {value}')

    '''
    # Initialize your model
    '''
    print("args.linear_dims", args.linear_dims)
    print("args.hidden_dim", args.hidden_dim)
    print("args.num_layers", args.num_layers)
    # print("args.fc_dims", args.fc_dims)

    model = RADFE(num_features=192, hidden_dim= args.hidden_dim, num_layers= args.num_layers, linear_dims= args.linear_dims, conv_channels = args.conv)
    
    model.to(device)

    if(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define an optimizer
    optimizer = Adam(model.parameters(), lr=params["learning_rate"])
    
    num_epochs = params['num_epochs'] # Set the number of epochs
    '''
        Code to train
    '''
    
    logging.info("[INFO] Training...")
    for epoch in (range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (radar_data, seg_map) in enumerate(tqdm(train_loader)):
            radar_data = radar_data.to(device).to(torch.float32)
            # print("radar_data_in_shape", radar_data.shape)
            seg_map = seg_map.to(device).to(torch.float32)
            optimizer.zero_grad()
      
            pred_seg_map = model(radar_data)
            #Compute the loss
            loss = bseg_binary_cross_entropy_loss(predicted_segmap=pred_seg_map, ground_truth_segmap=seg_map)
            running_loss += loss.item()
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
         

        if epoch % args.checkpoint_epoch == 0:
            # Save the model checkpoint
            checkpoint_model_name_ = checkpoint_model_name_folder + "model_epoch_"+str(epoch)  +str(args.linear_dims) + "_" + \
            str(args.hidden_dim) + "_" + str(args.num_layers) + "_" + "conv_"+ str(args.conv) + "_loss_" + str(format(running_loss,".2f")) + ".pth"
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_path=checkpoint_model_name_)

        # Print statistics
        logging.info(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        
        mse_error, dice_coef, chamfer_dist = evaluate_model_advanced(model, test_loader, device=device, num_classes=params["num_labels"], batch_size = params["batch_size"])

        
        logging.info(f"Epoch {epoch+1}, Accuracy Metrics, Pixel wise MSE: {mse_error}, Dice coefficient: {dice_coef}, Chamfer Distance: {chamfer_dist}")
        print(f"Epoch {epoch+1}, Accuracy Metrics, Pixel wise MSE: {mse_error}, Dice coefficient: {dice_coef}, Chamfer Distance: {chamfer_dist}")
        

    #Code to Evaluate
    '''
        Precision, Recall and F1 score are not suitable metrics for sole Radar based Object Detections using this approach
        0. Here our GT labels are binary masks as we are not annotating the dataset by hand and not using accurate calibration matrices for Camera to Radar Mapping
        1. Further, in Radar we have low angular precision due to lesser RX Antennas compared to imager.
        2. This low angular resolution results in mixing of smaller objects with Larger objects resulting in imperfect evaluation of Recall and Precision thus F1 scores
    '''
    mse_error, dice_coef, chamfer_dist = evaluate_model_advanced(model, test_loader, device=device, num_classes=params["num_labels"], batch_size = params["batch_size"])
    logging.info(f"Accuracy Metrics, Pixel wise MSE: {mse_error}, Dice coefficient: {dice_coef}, Chamfer Distance: {chamfer_dist}")
    print(f"Accuracy Metrics, Pixel wise MSE: {mse_error}, Dice coefficient: {dice_coef}, Chamfer Distance: {chamfer_dist}")
    
    '''
    #save the final model
    checkpoint_model_name_ = checkpoint_model_name_folder + "model_loss_" + str(format(running_loss, ".2f"))+ ".pth" #"_epoch_"+str(epoch) +str(args.linear_dims) + "_" + str(args.hidden_dim) + "_" + str(args.num_layers) + "_" + str(args.fc_dims) + "_" + str(args.bits) + "_" + str(args.noise_level) + ".pth"
    save_checkpoint(model, optimizer, epoch, loss, checkpoint_path=checkpoint_model_name_)
    '''

if(__name__ == "__main__"):
    #python evaluate_recall_small.py --linear_dims ${LINEAR_DIMS[$i]} $((${LINEAR_DIMS[$i]} * 2)) --hidden_dim ${HIDDEN_DIM[$j]} --num_layers 1 --conv ${CONV[$k]} --fc_dims $((2 * ${HIDDEN_DIM[$j]})) 28224
    #LINEAR_DIMS=(4 )
    #HIDDEN_DIM=(1024)
    #CONV=(32) #32 64 )
    parser = argparse.ArgumentParser()
    ## add a array with models layer sizes 
    parser.add_argument('--params_filename', type = str, default = "params.json")
    parser.add_argument('--load_checkpoint_path', type = str, default = "saved_checkpoint/radar_RAD_FFT__epoch_299[4, 8]_1024_1_conv_32_loss_14.008334251120687.pth")
    parser.add_argument('--save_checkpoint_path', type = str, default = "checkpoint/")

    parser.add_argument('--linear_dims', nargs='+', type=int, default = [4,8])
    #hidden dime 
    parser.add_argument('--hidden_dim', type=int, default= 1024)
    #num_layers
    parser.add_argument('--num_layers', type=int, default = 1)
    #conv_channels
    parser.add_argument('--conv', type=int, default=32)
    
    parser.add_argument('--checkpoint_epoch', type=int, default= 40)
    parser.add_argument('--bits', type=int, default=None )
    parser.add_argument('--noise_level', type=float, default=None)
    parser.add_argument('--load_checkpoint', type = bool, default = False)
    # parser.add_argument('--num_heads', type = int, default = 1)
    
    args = parser.parse_args()
    train(args)