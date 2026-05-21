import torchvision.datasets as datasets

datasets.CIFAR10(root="./data", train=True, download=True)
datasets.CIFAR10(root="./data", train=False, download=True)

datasets.SVHN(root='./data', split='train', download=True)
datasets.SVHN(root='./data', split='test', download=True)
