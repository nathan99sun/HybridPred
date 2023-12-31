Overview
========

The following is the code package for "A Physics-Informed Model Predicting Cycle Life of Lithium Ion Batteries," sponsored by Toyota and the Instutitue of Pure and Applied Mathematics under the RIPS 2023 REU.

<div align="center">

## Description

Accurately measuring the cycle lifetime of a commercial lithium-ion battery is crucial for performance and technology development in electric vehicles. In this project we introduce a novel approach combining a physics-based equation with a self-attention model to predict the cycle lifetimes of commercial lithium iron phosphate graphite cells via early-cycle data. Fitting capacity curves to this physics-based equation, we then use a self-attention layer to reconstruct entire battery capacity curves. Our model exhibits comparable performances to existing models while predicting the entire capacity curve instead of cycle life. This combines the advantages of data-driven architectures with physics-informed features to give a more complete description of the behavior of complex battery systems.

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


Nicolae, Daniel, Sara Sameer, Nathan Sun, Karena Yan. 2023. HybridPred package. Code and data repository at https://github.com/nathan99sun/HybridPred.

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

We benefited from Mattherix/template-python-package
