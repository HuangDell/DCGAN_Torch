import shutil
import torch
import torch.optim as optim
import os
from model.model import *

# log information
project_name = "GAN"
version = "v1.1"
path_name = "out/" + project_name + version + "/"
image_path_name = path_name+"images/"

# model
generator_features = 64
discriminator_features = 64
generator_input = 100


# model saving
save_name = path_name + project_name + ".pt"

# model parameters
criterion = nn.BCELoss()  # 选择交叉熵作为损失函数
optimizerD = optim.Adam
optimizerG = optim.Adam

# parameters
epochs = 50
lr = 0.0002
max_lr = 0.0002
grad_clip = 0.1
weight_decay = 1e-4
beta = 0.5

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data
IMAGE_SIZE = 128
DATA_PATH = './data/Standford_Dog_Dataset/images/Images'
BATCH_SIZE = 1

if not os.path.exists(path_name):
    os.makedirs(path_name)
    os.makedirs(image_path_name)
    shutil.copy("config.py", path_name + "model.config")  # 将该模型对应的配置信息保存
