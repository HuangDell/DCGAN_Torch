import torch
import torchvision
import torchvision.transforms as transforms
import config

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.CIFAR10(root=config.DATA_PATH, train=True, download=False, transform=transform_train)
train_data = torch.utils.data.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                                         num_workers=2)  # 2线程读取数据
test_set = torchvision.datasets.CIFAR10(root=config.DATA_PATH, train=False, download=False, transform=transform_test)
test_data = torch.utils.data.DataLoader(test_set, batch_size=2 * config.BATCH_SIZE, shuffle=False, num_workers=2)
