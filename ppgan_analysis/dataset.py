import torchvision.datasets as datasets
from torchvision import transforms

def get_dataset(dataset_name, dataset_type):
    train = dataset_type=="train"
    if dataset_name == "mnist":
        dataset = datasets.MNIST("/tmp",train=train,download=True,transform=transforms.ToTensor())
        return dataset
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10("/tmp",train=train,download=True,transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
            ])
        )
        return dataset
    elif dataset_name == "celeba":
        dataset = datasets.CelebA("~/research/ppgan_project/ppgan_attacks/data",split='train',download=False, transform=transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ]))
        return dataset
    else:
        raise NotImplementedError