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
Our model is built with EfficientNetV2-M for images and Bag of Words for the text which are concatenated to two fully-connected layers.

### Preprocesor.py
**Image Preprocessing**
During training we apply vast amounts of transformations to avoid overfitting and generalize our models as well as possible. 

### Garbage_classification.py
### Statistics Folder

**Our steps**


