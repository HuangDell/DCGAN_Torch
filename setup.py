import torch.cuda
import torch.nn as nn
from data import load
from tqdm import tqdm
from util.output import *
from model.model import *


def _train(model, train_data, test_data):
    torch.cuda.empty_cache()
    optimizer = config.optimizer(model.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.max_lr, epochs=config.epochs,
                                                    steps_per_epoch=len(train_data))

    loss_list, acc_list = [], []
    bar = tqdm(range(config.epochs))
    for _ in bar:
        model.train()
        losses = 0.0
        for batch in train_data:
            loss = model.train_step(batch)
            loss.backward()
            losses += loss.item()

            nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        acc = model.evaluate(test_data)
        result = f'loss {losses / len(train_data)},acc {acc}'
        loss_list.append(losses)
        acc_list.append(acc)
        bar.set_description(result)
        printf.info(result)
    return loss_list, acc_list


def test():
    model = config.model()
    model.load()
    acc = model.evaluate(load.test_data)
    printf.debug(f"Accuracy {acc*100}%")


def train():
    model = ResNet9()
    loss, acc = _train(model, test_data=load.test_data, train_data=load.train_data)
    save_model(model)
    draw_result(config.epochs, loss, acc)
