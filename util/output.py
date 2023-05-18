import os
import torch
import config
import matplotlib.pyplot as plt
import shutil
from util.logger import printf


def create_model_dir():
    os.mkdir(config.path_name)  # 为每个模型创建一个单独的项目保存目录
    os.chdir(config.path_name)  # 将工作目录调整为模型对于的目录


def save_model(**models):
    for idx, (name, model) in enumerate(models.items()):
        torch.save(model.state_dict(), config.project_name + "_" + name + ".pt")
        printf.info(f"{config.project_name + config.version} save successfully.")
    shutil.copy("../../config.py", "config.py")  # 将该模型对应的配置信息保存


def draw_result(epoch, **features):
    plt.title(config.project_name + config.version)
    for idx, (name, feature) in enumerate(features.items()):
        plt.plot(range(epoch), feature, label=name)
    plt.legend()
    plt.savefig(config.save_name)


def visualize(index=0, **images):
    n_images = len(images)
    # plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.savefig(config.image_path_name + f"result{index}.jpg")
