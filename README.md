# ChirpNet: Noise-Resilient Sequential Chirp-based Radar Processing for Object Detection

This repository contains the code for the paper **"ChirpNet: Noise-Resilient Sequential Chirp based Radar Processing for Object Detection"** by Sudarshan Sharma\*, Hemant Kumawat\*, and Saibal Mukhopadhyay from the School of Electrical and Computer Engineering, Georgia Institute of Technology, which appeared in IEEE IMS'24. [Link](https://ieeexplore.ieee.org/document/10600387)

\* - Equal Contributions

## Abstract

Radar-based object detection (OD) requires extensive pre-processing and complex Machine Learning (ML) pipelines. Previous approaches have attempted to address these challenges by processing raw radar data frames directly from the ADC or through FFT-based post-processing. However, the input data requirements and model complexity continue to impose significant computational overhead on the edge system. In this work, we introduce ChirpNet, a noise-resilient and efficient radar processing ML architecture for object detection. Diverging from previous approaches, we directly handle raw ADC data from multiple antennas per chirp using a sequential model, resulting in a substantial 15× reduction in complexity and a 3× reduction in latency, while maintaining competitive OD performance. Furthermore, our proposed scheme is robust to input noise variations compared to prior works.



## Setup Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/sudo-sh/ChirpNet.git
    cd ChirpNet
    ```

2. **Create and activate the conda environment**:
    ```sh
    conda env create -f environment.yml
    conda activate chirpnet
    ```

3. **Prepare the dataset**:
    - Download the `data/` `saved_dl/` and `saved_checkpoint` folders from dropbox [Link](https://www.dropbox.com/scl/fo/7f7zusk5st1qbkuw56jpm/AMYx2xXQ0VQ0rKACWodFCCY?rlkey=gdipvo2u1usiyqhid42ais8tt&dl=0)
    - Place your dataset in the `data/` directory.
    - Modify the data loading scripts (`h5_dataloader.py`, `chirpnet_dataloader.py`) if necessary to match your dataset structure.

4. **Train the model**:
    ```sh
    python train_and_eval.py --load_checkpoint True
    ```

## Usage

### Training
To train the model, run the `train_and_eval.py` script. Ensure that your dataset is properly configured and available in the `data/` directory.

### Evaluation
To evaluate the model, modify and use the provided scripts according to your evaluation protocols. For example, you can save model checkpoints during training and load them later for evaluation.

## Discussions

## Evaluation Metrics

### Precision, Recall, and F1 Score

The typical metrics of Precision, Recall, and F1 score are not entirely suitable for evaluating sole radar-based object detection using this approach for several reasons:

1. **Ground Truth Labels**:
   - Our ground truth (GT) labels are binary masks in the Image space. We generate these masks automatically using Images rather than through manual annotation, and we do not use precise calibration matrices for camera-to-radar mapping.

2. **Angular Precision**:
   - Radar sensors, compared to imaging sensors, have lower angular precision due to the fewer number of receiver (RX) antennas. This limitation affects the accuracy of object localization in the angular domain.

3. **Object Mixing**:
   - The low angular resolution can cause smaller objects to be merged with nearby larger objects in the image space. This merging leads to imperfect evaluations of Recall and Precision, resulting in less reliable F1 scores.

Given these challenges, while we report Recall for completeness in the paper, we recommend interpreting these metrics with caution and considering alternative evaluation approaches better suited to radar data.

### Recommendations for Evaluation

- **Dice Coefficient**: The Dice coefficient is a useful metric for evaluating the similarity between the predicted and ground truth binary masks. Higher values indicate better performance.

- **Chamfer Distance**: Chamfer distance measures the similarity between the predicted and ground truth point sets. It is particularly useful in scenarios where precise alignment between predicted and ground truth masks is critical.

- **Pixel-wise Mean Squared Error (MSE)**: MSE is a standard metric for measuring the average squared difference between predicted and ground truth pixel values. It provides a quantitative measure of prediction accuracy.

We welcome feedback and suggestions from the community to improve the evaluation methodologies for radar-based object detection.


## Citation

If you find this code useful in your research, please consider citing:
```
@INPROCEEDINGS{10600387,
  author={Sharma, Sudarshan and Kumawat, Hemant and Mukhopadhyay, Saibal},
  booktitle={2024 IEEE/MTT-S International Microwave Symposium - IMS 2024}, 
  title={ChirpNet: Noise-Resilient Sequential Chirp Based Radar Processing for Object Detection}, 
  year={2024},
  volume={},
  number={},
  pages={102-105},
  keywords={Chirp;Computational modeling;Noise;Radar detection;Object detection;Radar;Radar antennas;Radar;Object detection;Efficient processing;Noise robustness},
  doi={10.1109/IMS40175.2024.10600387}}
```