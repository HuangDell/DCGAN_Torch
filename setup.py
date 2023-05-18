import torch.cuda
import torch.nn as nn

import config
from data import load
from tqdm import tqdm
from util.output import *
from model.model import *


def _train(G, D, train_data, test_data):
    torch.cuda.empty_cache()
    G.apply(weights_init)
    D.apply(weights_init)

    fixed_noise = torch.randn(64, config.generator_input, 1, 1, device=config.device)

    optimizerG = config.optimizerG(G.parameters(), config.lr, weight_decay=config.weight_decay,
                                   betas=(config.beta, 0.999))
    optimizerD = config.optimizerD(D.parameters(), config.lr, weight_decay=config.weight_decay,
                                   betas=(config.beta, 0.999))

    schedulerG = torch.optim.lr_scheduler.OneCycleLR(optimizerG, config.max_lr, epochs=config.epochs,
                                                     steps_per_epoch=len(train_data))

    schedulerD = torch.optim.lr_scheduler.OneCycleLR(optimizerD, config.max_lr, epochs=config.epochs,
                                                     steps_per_epoch=len(train_data))

    G_loss_list, D_loss_list = [], []

    bar = tqdm(range(config.epochs))
    for _ in bar:
        for batch in train_data:
            # 首先用全真数据 训练D 网络
            D.zero_grad()
            real_data = batch[0].to(config.device)
            b_size = real_data.size(0)

            label = torch.full((b_size,),1, dtype=torch.float, device=config.device)

            output = D(real_data).view(-1)
            lossD_real = config.criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            # 全假数据训练D网络
            noise = torch.randn(b_size, config.generator_input, 1, 1, device=config.device)
            fake_data = G(noise)
            label.fill_(0)

            output = D(fake_data.detach()).view(-1)
            lossD_fake = config.criterion(output,label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()

            lossD = lossD_fake+lossD_real

            optimizerD.step()

            # 下面开始训练G网络
            G.zero_grad()
            label.fill_(1)
            output = D(fake_data).view(-1)
            lossG = config.criterion(output,label)
            lossG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()
        result = f'Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}, D(x):{D_x:.4f}, D(G(x)): {D_G_z1:.4f}/{D_G_z2:.4f}'
        G_loss_list.append(lossG.item())
        D_loss_list.append(lossD.item())
        bar.set_description(result)
        printf.info(result)
    return G_loss_list, D_loss_list


def test():
    model = config.model()
    model.load()
    acc = model.evaluate(load.test_data)
    printf.debug(f"Accuracy {acc * 100}%")


def train():
    G = Generator().to(config.device)
    D = Discriminator().to(config.device)
    G_loss, D_loss = _train(G, D, test_data=load.test_data, train_data=load.train_data)
    save_model(G=G,D=D)
    draw_result(config.epochs, G_Loss=G_loss, D_Loss=D_loss)
