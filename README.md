# odenet_crowd
This repository contains codes for producing synthetic data, training the model and producing all figures in the article
https://arxiv.org/abs/2210.09602

Process 1: produce the simulation data
The codes for generating synthetic data with SFM and ORCA are in the folder "sfm_data" and "orca_data" respectively. 

Process 2: train the ODE-Net
Training data from the SFM and ORCA are provided in the folder "result/sfm" and "result/orca" respectively. Parameters of the neural network are saved as the trained model.


Process 3: test the ODE-Net
Run "draw_sfm.py" or "draw_orca.py" to draw figures and animations of the test result of the trained ODE-Net.


2022/10/20
