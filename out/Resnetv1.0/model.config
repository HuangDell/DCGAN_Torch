import shutil
import torch
import torch.optim as optim
import os
from model.model import *

# log information
project_name = "Resnet"
version = "v1.0"
path_name = "out/" + project_name + version + "/"

# model
model = ResNet9

# model saving
save_name = path_name + project_name + ".pt"

# model parameters
criterion = nn.CrossEntropyLoss()  # 选择交叉熵作为损失函数
optimizer = optim.Adam

# parameters
epochs = 50
lr = 0.01
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data
DATA_PATH = './data'
BATCH_SIZE = 400

if not os.path.exists(path_name):
    os.makedirs(path_name)
    shutil.copy("config.py", path_name + "model.config")  # 将该模型对应的配置信息保存

