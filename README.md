# An Implementation of Ensemble Learning using Transformers and Convolutional Networks for Masked Face Recognition
This repository contains the code implementation of the paper **[Ensemble Learning using Transformers and Convolutional Networks for Masked Face Recognition](https://arxiv.org/abs/2210.04816)** by Al-Sinan et al. , and some further exploration for improvements.

## Overview
Wearing a face mask was one of the changes we had to make to slow the spread of the coronavirus. The constant covering of our faces with masks has created a need to understand and investigate how this behavior affects the recognition capability of face recognition systems. When dealing with unconstrained general face recognition cases, current face recognition systems have extremely high accuracy, but they do not generalize well with occluded masked faces. The proposed method combines two popular deep learning techniques, Convolutional Neural Networks (CNNs) and Transformers, to produce an ensemble model capable of accurately recognizing masked faces. Two CNN model, fine-tuned on FaceNet pre-trained models extracting features from the face image, with two Transformer model learning the relationship between these features and the person's identity were combined in the ensemble using majory voting or average weighted method. Experiment results showed that the proposed ensemble model outperforms existing state-of-the-art masked face recognition methods with a 92% accuracy.

## Dataset and Pre-trained Models
The proposed system was tested on a synthetically masked LFW dataset generated in this study. Moreover, the MAFA and RFMR datasets were used to verify the robustness of the proposed methodology.
Please use these links to download the dataset and pre-trained models:
1. LFW dataset and pre-trained models:  https://drive.google.com/drive/folders/1uiTAWdU2DMyy27j3H6BPnLCiQvFRH5nD
2. MAFA dataset: https://drive.google.com/drive/folders/1q0UwRZsNGuPtoUMFrP-U06DSYp_2E1wB


## Modelling
### CNN models fine-tuned on FaceNet dataset 
Three CNN models fine-tuned on different pre-trained models (VGG16, EfficientNet, FaceNet) were studied in the original study. It was seen that the CNN models fine-tuned on FaceNet pre-trained model provided the best accuracy and hence, this model was selected for the for the final classification task. Since, FaceNet was trained only on images of maskless people and the aim of the project is to detect images of masked faces, the last layers of this model were modified and a dropout and batch normalization layers were added to this model. The output features of this model were fed into the classification layer. A Softmax function in the classification layer was used, which assigns a probability for each predicted subject.

![Alt text](cnn_models.png?raw=true)

### Transformer models 
A Transformer model, using only the encoder component was also used for the masked face recognition task. The proposed Transformer, which was inspired by the work of [Dosovitskiy et al](https://arxiv.org/abs/2010.11929), which accepts an input image and splits it into a number of patches. Each patch represents one time step and is fed into the encoder. However, the spatial information of the input image is not preserved by this image segmentation. To address this issue, the study by Al-Sinan et al. modifies transformers, where each patch uses position embedding to encode its position in the original input image. The resulting patch with its embedded position was fed into the encoder of the Transformer. The encoder is made up of a layer normalization unit and a multi-head attention (MHA) unit. A residual connection was also used with the MHA unit.

![Alt text](transformer.png?raw=true)

### An Ensemble of two CNN and two Transformer models
Finally, in the paper, the 4 individual models (two FaceNet pre-trained models and 2 Transformers) were combined using average weighted ensembling technique. 
![Alt text](ensemble.png?raw=true)

## Improvements from our implementation:

1.  <b>Identifying best weights for each model:<\b> The original implementation involves majority voting involving equal weights for every model to create an ensemble. In order to explore improvements, we implemented the grid search to identify the optimal weights for every model before combining them. We found the following to be the optimum weights, with an accuracy of 92.015\% (similar to the original contribution). 
  
  #### Optimum weights found using grid search
 | FaceNet 1 | FaceNet 2| Transformer 1 | Transformer 2 
| -- | -- | -- | -- |
0.25 | 0.25 | 0.28 | 0.22 |

2. <b>Using stacking ensemble techniques:</b> We used another approach for creating an ensemble of the
originally proposed best-performing individual models, known as a stacked generalization or stacking.
Stacked generalization is an ensemble method in which a new model discovers the optimal way to combine
the predictions of various existing models. Here, we have used Logistic Regression from the scikit-learn
library as a meta-learner. The observed accuracy was higher compared to the original by a significant
improvement gain of 2.2%, resulting in a final accuracy of 94.22%


## Results

| Model | Top-1 Accuracy | Top-5 Accuracy
| -- | -- | -- |
CNN-VGG16 | 73.38% | 82.05% |
CNN-EfficientNet | 79.61% | 84.41% |
CNN-FaceNet | 80.30% | 85.24% |
Transformer | 69.04% | 78.70% |
Ensemble Learning (Average weighted) | **92.01%** | **96.57%** |
Ensemble Learning (Grid search weighted) | **92.01%** | **96.81%** |
Ensemble Learning (Stacked) | **94.22%** | **98.12%** |
