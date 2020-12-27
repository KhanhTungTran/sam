import torch
import torchvision
from data.dataclass import REST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = REST(root='./data', train=True, transform=train_transform)
        test_set = REST(root='./data', train=False, transform=test_transform)
        # train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform)
        # test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        print(self.train, train_set)

        self.classes = ('bathroom', 'bedroom', 'dining_room', 'exterior', 'interior', 'kitchen', 'living_room')

    def _get_statistics(self):
        statistics_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        train_set = REST(root='./rest', train=True, transform=statistics_transform)
        test_set = REST(root='./rest', train=False, transform=statistics_transform)
        # train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, transform=transforms.ToTensor())
        # test_set = torchvision.datasets.CIFAR10(root='./cifar', train=False, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)] + [d[0] for d in DataLoader(test_set)])
        # print(data)
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])