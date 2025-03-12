from torchvision import datasets
from torchvision import transforms
import torch
# import crypten.communicator as comm
import crypten
def get_dataset(dataset_info, dataset_type, secure=False):
    train = dataset_type=="train"
    dataset_name, dataset_model = dataset_info.split("_")
    # print("dataset name",dataset_name)
    # assert False
    if secure==False:
        if dataset_name == "mnist":
            if dataset_model == "fc":
                dataset = datasets.MNIST("/tmp",train=train,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5,0.5)]))#transforms.Normalize(0.5,0.5)
            elif dataset_model == "cnn":
                dataset = datasets.MNIST("/tmp",train=train,download=True,transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
            return dataset
        elif dataset_name == "cifar10":
            # print("cifar10")
            dataset = datasets.CIFAR10("/tmp",train=train,download=True,transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                ])
            )
            return dataset
        elif dataset_name == "celeba":
            dataset = datasets.CelebA("/home/tson1997/research/ppgan_project/ppgan_attacks/data",split='train',download=False, transform=transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ]))
            return dataset
        else:
            raise NotImplementedError

    else:
        if dataset_name == "mnist":
            dataset = MPC_MNIST_Dataset("/tmp",train=train,download=True,transform=transforms.ToTensor())
            return dataset

def get_noise(args):
    # if args.model_type == "cnn":
        # return torch.randn(args.batch_size, args.zdim, 1, 1).to(args.device)
    # elif args.model_type == "fc":
    z = torch.randn(args.batch_size, args.zdim).to(args.device)
    return z.reshape(z.shape[0],z.shape[1],1,1)

class MPC_MNIST_Dataset(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return crypten.cryptensor(img), label
