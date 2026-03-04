# Assignment02
Garbage Classification Model - Programming Assignment

This repository contains our solution for the Garbage Classification Assignmnent. 

We include our modular code for preprocessing, training and evaluation, the best model we trained as well as diagrams and metrics for that model. 

Our model is a multi-modal approach, combining both the text description of the garbage item as well as its image to classify it into the four different classes. 

The classes are the following: 

**Black** – Landfill Waste 
**Blue** – Recycling
**Green** – Organics
**TTR** – Residual Trash

## Files
### Model.py
Our model is built with EfficientNetV2-M for images and Bag of Words for the text which are concatenated to two fully-connected layers. We have also experimented with EfficientNetV2-S, but achieved better results with the M variant.

### Preprocesor.py
This file implements the data preprocessing and dataset loading pipeline for the multimodal garbage classification model. It applies image augmentations during training, extracts text features from image filenames and converts them into bag-of-words vectors. The module also provides utilities for building the vocabulary, counting dataset images and computing class weights for imbalanced data. 


### Garbage_classification.py
This script implements the full training pipeline for the multimodal garbage classification model.

To handle data imbalance, we use: 

- Inverse frequency weighting to penalize minority classes with higher loss 

- Label smoothing using soft targets (0.1 smoothing) instead of hard one-shot 

- Monitoring F1-scores (weighted & macro), not just accuracy 

- Different augmentation techniques to prevent overfitting and improve generalization 

- Progressive fine tuning by unfreezing blocks of the model progressively and see if the performance improved 

- Dropout to the text input and text modality to reduce overreliance on text when training.

### Outputs Folder
The output folder contains the confusion matrix, examples of misclassified images, a csv file of our test predictions, as well as statistics for our different training stages. 
We chose the model that is represented in stage_7_unfreeze_6_blocks_curves.png .



