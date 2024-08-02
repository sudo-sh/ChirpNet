import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load a model checkpoint.

    Parameters:
    checkpoint_path (str): File path of the checkpoint.
    model (torch.nn.Module): The model to load the checkpoint into.
    optimizer (torch.optim.Optimizer): The optimizer to load the checkpoint into.

    Returns:
    int, float: The epoch and loss from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss



def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save a model checkpoint.

    Parameters:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    epoch (int): The current epoch number.
    loss (float): The loss at the current epoch.
    checkpoint_path (str): File path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)


def visualize_and_save_radar(y_true_batch, y_pred_batch, n, save_path):
    """
    Visualize and save n pairs of ground truth and predicted masks side by side.

    Parameters:
    y_true_batch (torch.Tensor): Batch of ground truth masks, shape (batch_size, height, width).
    y_pred_batch (torch.Tensor): Batch of predicted masks, shape (batch_size, 1, height, width).
    n (int): Number of image pairs to plot.
    save_path (str): Path to save the visualization.
    """
    # Ensure n does not exceed the batch size
    # n = min(n, y_true_batch.size(0))

    if(n==1):
        # Set up the plot
        fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
        y_true = y_true_batch[0].cpu().numpy()
        axes[0].imshow(y_true, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Predicted mask
        y_pred = torch.sigmoid(y_pred_batch[0, 0]).cpu().numpy()  # Squeeze the channel dimension
        # y_pred = torch.sigmoid(y_pred_batch[i, 0]).cpu().numpy()  # Squeeze the channel dimension
        # print("Sigmoid",y_pred)
        y_pred = (y_pred > 0.5).astype(np.int64)  # Apply threshold and convert to integer
        # print("Threshold", y_pred)
        # exit()
        axes[1].imshow(y_pred, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
    else:
        
        # Set up the plot
        fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

        for i in range(n):
            # Ground truth
            y_true = y_true_batch[i].cpu().numpy()
            axes[i, 0].imshow(y_true, cmap='gray')
            axes[i, 0].set_title(f'Ground Truth {i+1}')
            axes[i, 0].axis('off')

            # Predicted mask
            y_pred = torch.sigmoid(y_pred_batch[i, 0]).cpu().numpy()  # Squeeze the channel dimension
            # y_pred = torch.sigmoid(y_pred_batch[i, 0]).cpu().numpy()  # Squeeze the channel dimension
            # print("Sigmoid",y_pred)
            y_pred = (y_pred > 0.5).astype(np.int64)  # Apply threshold and convert to integer
            # print("Threshold", y_pred)
            # exit()
            axes[i, 1].imshow(y_pred, cmap='gray')
            axes[i, 1].set_title(f'Predicted Mask {i+1}')
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)


def visualize_and_save(batch_index, input_images, y_true_batch, y_pred_batch, save_path):
    """
    Visualize and save the input image, ground truth mask, and predicted mask side by side.

    Parameters:
    batch_index (int): Index of the batch to visualize.
    input_images (numpy.ndarray): Batch of input images, shape (batch_size, height, width, channels).
    y_true_batch (numpy.ndarray): Batch of ground truth masks, shape (batch_size, height, width).
    y_pred_batch (numpy.ndarray): Batch of predicted masks, shape (batch_size, 1, height, width).
    save_path (str): Path to save the visualization.
    """
    input_image = input_images[batch_index].cpu().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))
    input_image = input_image.astype(np.uint8)

    y_true = 255*y_true_batch[batch_index].cpu().numpy()
    y_pred = 255*torch.sigmoid(y_pred_batch[batch_index, 0].cpu())  # Squeeze the channel dimension

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(y_true, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(y_pred, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)

def dice_coefficient_batch(predicted_batch, ground_truth_batch):
    """
    Compute the Dice Coefficient for a batch of predictions and ground truths.

    :param predicted_batch: A batch of predicted masks (binary) as a numpy array.
    :param ground_truth_batch: A batch of ground truth masks (binary) as a numpy array.
    :return: A list of Dice Coefficients for each pair in the batch, and the average Dice Coefficient for the batch.
    """
    dice_scores = []

    # Iterating over each pair of predicted and ground truth masks
    for predicted, ground_truth in zip(predicted_batch, ground_truth_batch):
        # Flattening the arrays
        predicted = predicted.flatten()
        ground_truth = ground_truth.flatten()

        # Calculating intersection and union
        intersection = np.sum(predicted * ground_truth)
        total = np.sum(predicted) + np.sum(ground_truth)

        # Handling division by zero
        if total == 0:
            dice_score = 1.0
        else:
            dice_score = 2 * intersection / total

        dice_scores.append(dice_score)

    # Calculating average dice coefficient
    avg_dice = np.mean(dice_scores)

    return sum(dice_scores)


def chamfer_distance_batch(gt_batch, binary_mask_batch):
    chamfer_distances = []
    
    # Iterate through the batch
    for gt, binary_mask in zip(gt_batch, binary_mask_batch):
        # Ensure the images are binary
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

        # Compute the distance transform
        dt_gt = distance_transform_edt(gt == 0)
        dt_mask = distance_transform_edt(binary_mask == 0)

        # Compute the Chamfer distance for the current pair of images
        chamfer_dist = np.mean(dt_gt[binary_mask > 0]) + np.mean(dt_mask[gt > 0])
        
        # Append the result to the list
        chamfer_distances.append(chamfer_dist)
    
    # Convert the list to a numpy array
    chamfer_distances = np.array(chamfer_distances)
    
    # Compute the mean Chamfer distance for the batch
    mean_chamfer_distance = np.mean(chamfer_distances)
    
    return mean_chamfer_distance

    
def calculate_pixelwise_mse(pred_segmap, gt_segmap):
    """
    Calculate the pixel-wise Mean Squared Error for binary segmentation.

    Parameters:
    pred_segmap (torch.Tensor): Predicted segmentation maps of shape (batch, 1, width, height)
    gt_segmap (torch.Tensor): Ground truth segmentation maps of shape (batch, width, height)

    Returns:
    float: Pixel-wise Mean Squared Error.
    """
    # Move tensors to CPU and convert to numpy
    pred_segmap = torch.sigmoid(pred_segmap.cpu()).numpy()
    gt_segmap = gt_segmap.cpu().numpy()

    # Apply sigmoid to the predicted map
    # pred_segmap = 1 / (1 + np.exp(-pred_segmap))  # Sigmoid function

    # Flatten the arrays
    pred_flat = pred_segmap.squeeze(1).reshape(-1)
    gt_flat = gt_segmap.reshape(-1)

    # Calculate MSE
    mse = np.mean((pred_flat - gt_flat) ** 2)

    return mse

def calculate_pixelwise_f1(pred_segmap, gt_segmap):
    """
    Calculate the pixel-wise F1 Score for binary segmentation.

    Parameters:
    pred_segmap (torch.Tensor): Predicted segmentation maps of shape (batch, 1, width, height)
    gt_segmap (torch.Tensor): Ground truth segmentation maps of shape (batch, width, height)

    Returns:
    float: Pixel-wise F1 Score.
    """
    # Move tensors to CPU and convert to numpy
    pred_segmap = pred_segmap.cpu().numpy()
    gt_segmap = gt_segmap.cpu().numpy()

    # Apply sigmoid to the predicted map and convert to binary (0 or 1)
    pred_segmap = 1 / (1 + np.exp(-pred_segmap))  # Sigmoid function
    pred_flat = (pred_segmap > 0.5).astype(np.int32).squeeze(1).reshape(-1)  # Threshold and reshape

    # Reshape the ground truth segmentation map
    gt_flat = gt_segmap.reshape(-1)

    # Calculate F1 Score
    f1 = f1_score(gt_flat, pred_flat, average='binary')

    return f1

def bseg_binary_cross_entropy_loss(predicted_segmap, ground_truth_segmap):
    """
    Compute the binary cross entropy loss.

    Parameters:
    predicted_segmap (torch.Tensor): The predicted segmentation map, 
                                     should have values between 0 and 1.
    ground_truth_segmap (torch.Tensor): The ground truth segmentation map, 
                                        should be a binary map (values of 0 or 1).

    Returns:
    torch.Tensor: The computed binary cross entropy loss.
    """

    # Ensure the predicted_segmap is in the form of probabilities
    predicted_segmap = torch.sigmoid(predicted_segmap)

    # Flatten the tensors to make them compatible for the BCE loss calculation
   
    predicted_segmap_flat = predicted_segmap.view(-1)
    ground_truth_segmap_flat = ground_truth_segmap.view(-1)
 
    # Calculate Binary Cross Entropy Loss
    loss = F.binary_cross_entropy(predicted_segmap_flat, ground_truth_segmap_flat)

    return loss





