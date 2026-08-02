# Histopathologic Cancer Detection

This repository contains my CSPB 3202 Homework 5 analysis for the Kaggle Histopathologic Cancer Detection competition.

The analysis trains a small convolutional neural network to predict whether the center of a pathology image patch contains tumor tissue. It compares a baseline model with a second model that uses image augmentation and a lower learning rate. The notebook also includes class counts, example images, a brightness comparison, validation results, and the code used to create the Kaggle submission file.

The competition data is not included because it is 7.76 GB. It can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data).

The completed Kaggle notebook is available [here](https://www.kaggle.com/code/bryced0924/hw5-histopathologic-cancer-detection).

To run the analysis on Kaggle, attach the Histopathologic Cancer Detection competition data and run the notebook from top to bottom with a GPU accelerator. The notebook writes `submission.csv` and the trained model to `/kaggle/working`.
