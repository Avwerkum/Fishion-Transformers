# This file is used to configure the training scripts.
# Adjust parameters and settings before running each script.

#Import two dependencies
import timm
import torch

# Models
ViTB16 =  timm.create_model('vit_base_patch16_224', pretrained=True)
efficientnet_b2 = timm.create_model('efficientnet_b2', pretrained=True)
deit_base_patch16_224 = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

#Set data locations
TRAIN_DIR = 
TEST_DIR = 
VAL_DIR = 

#Set save location
SAVE_DIR =

# Adjust 
BATCH_SIZE = 32
NUM_WORKERS = 0

NUM_CLASSES = 30

#Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.03
NUM_EPOCHS = 100

#CosineAnnealing
T_MAX= 10
ETA_MIN = 1e-6

#Early Stopping Criteria
PATIENCE = 10
DELTA = 0.001
