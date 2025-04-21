Overview
========

The following is the code package for _Optimizing Cycle Life Prediction of Lithium-ion Batteries via a Physics-Informed Model_, sponsored by Toyota and the Instutitue of Pure and Applied Mathematics under the RIPS 2023 REU.

<div align="center">

## Description

Accurately measuring the cycle lifetime of commercial lithium-ion batteries is crucial for performance and technology development. We introduce a novel hybrid approach combining a physics-based equation with a self-attention model to predict the cycle lifetimes of commercial lithium iron phosphate graphite cells via early-cycle data. After fitting capacity loss curves to this physics-based equation, we then use a self-attention layer to reconstruct entire battery capacity loss curves. Our model exhibits comparable performances to existing models while predicting more information: the entire capacity loss curve instead of cycle life. This provides more robustness and interpretability: our model does not need to be retrained for a different notion of end-of-life and is backed by physical intuition.

## Overview

This repository contains the following folders: autoencoder, data, physics, cleaning, misc, source, and transformer. The cleaning folder contains code to clean raw data. The resulting cleaned data, along with the original data, is stored in the data folder. The source folder contains code used in our baseline models for replicating prior results. The autoencoder folder contains files used in development of an elastic net/autoencoder model to predict cycle life from early-cycle time series. The transformer folder contains a basic implementation of a transformer designed to regress cycle life from early-cycle time series, while the physics folder contains code that implements a hybrid physics/data-driven method to predict entire capacity curves from early-cycle statistics data. Finally, the misc folder contains any miscellaneous files used in our project.

## Technology Used

<div>
  <img name = "Python" src = "https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white">
   <img name = "Pandas" src = "https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
  <img name = "Scikit-learn" src = "https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <br>
  <img name = "Tensorflow" src = "https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
   <img name = "Numpy" src = "https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white">
</div>

## Usage example


Nicolae, Daniel, Sara Sameer, Nathan Sun, Karena Yan. _Optimizing Cycle Life Prediction of Lithium-ion Batteries via a Physics-Informed Model_, to appear in Transactions on Machine Learning Research (2025).

## Contributors


  <br>
  Daniel Nicolae
  </br>
   Sara Sameer
  <br>
  Nathan Sun
  <br>
   Karena Yan
      
## Acknowledgements

We benefited from Mattherix/template-python-package. 
