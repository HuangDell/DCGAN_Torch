import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import config

transform_train = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = ImageFolder(root=config.DATA_PATH, transform=transform_train)
test_set = ImageFolder(root=config.DATA_PATH, transform=transform_train)

train_data = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                                         num_workers=4)  # 2线程读取数据

test_data = torch.utils.data.DataLoader(test_set, batch_size=2 * config.BATCH_SIZE, shuffle=False, num_workers=4)
