import os
import torch
import config
import matplotlib.pyplot as plt
import shutil
from util.logger import printf


def create_model_dir():
    os.mkdir(config.path_name)  # 为每个模型创建一个单独的项目保存目录
    os.chdir(config.path_name)  # 将工作目录调整为模型对于的目录


def save_model(model):
    torch.save(model.state_dict(), config.project_name + ".pt")
    shutil.copy("../../config.py", "config.py")  # 将该模型对应的配置信息保存
    printf.info(f"{config.project_name+config.version} save successfully.")


def echo_epoch_end(epoch, result):
    printf("Epoch [{}], lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".
           format(epoch, result['lr'][-1], result['train_loss'], result['test_loss'], result['test_acc']))


def draw_result(epoch, loss, acc):
    plt.title(config.project_name+config.version)
    plt.plot(range(epoch), loss, label='Train loss')
    plt.plot(range(epoch), acc, label='Test acc')
    plt.legend()
    plt.savefig(config.save_name)
