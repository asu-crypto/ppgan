from typing import Counter
from options import *
from model import *
from reconstruct_data import reconstruct_data
from util import *
from dataset import get_dataset
from torch.utils.data import DataLoader
from reconstruct_model import recon_model
import torchvision

args=options().parse_args()
seed_everything(args.seed)
model_generator = ModelGenerator()

mtype="fc" if args.dataset=="mnist" else "dc"
if args.dataset == "mnist":
    model = model_generator.get_model("mlp",args.num_secure,784,1024,512,256,1).to("cuda")
    gen = model_generator.get_model("mlp",0, 100,256,512,1024,784).to("cuda")
elif args.dataset == "cifar10":
    model = model_generator.get_model("cnn", args.num_secure, (3,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
    gen = DCGenerator().to("cuda")
# disc = model_generator.get_model("cnn",args.num_secure, (1,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
print("Number of parameters to reconstruct:", weight_count(model))

dataset = get_dataset(args.dataset,"train")
dataloader = DataLoader(dataset, batch_size=args.batch_size_real)
 
def l2loss(a,b):
    return ((a-b).square().mean()).sqrt()



# weight recovery attack
# sample_input = torch.rand(args.batch_size_fake, 3,64,64).to("cuda")
# sample_output = model(sample_input)
# print(sample_input.shape, sample_output.shape)
# assert False
# model_estimate = recon_model(sample_input, sample_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, "mlp",args.num_secure,784,1024,512,256,1)
# # model_estimate = recon_model(sample_input, sample_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, "cnn",args.num_secure, (1,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0))

# estimated = list(model_estimate.parameters())
# actual = list(disc.parameters())

# print("BENCHMARK DISTANCE BETWEEN MODEL AND ITS RECOVERY:")
# print("Similarity Distance:",sim_dist(estimated, actual))
# print("MAE Distance:",mae_dist(estimated, actual))
# print("MSE Distance:",mse_dist(estimated, actual))

# data recovery attack
# sample_input = torch.rand(args.batch_size_real, 1, 784).to("cuda")
# if args.dataset == "mnist":
#     dataset = torchvision.datasets.MNIST(root="/tmp",train=True,download=True, transform=torchvision.transforms.ToTensor())
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_real)
# elif args.dataset =="cifar10":
#     dataset = torchvision.datasets.CIFAR10(root="/tmp",train=True,download=True, transform=torchvision.transforms.ToTensor())
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_real)
counter=0

optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

z_fixed = torch.randn(args.batch_size_real, 100, 1, 1).to("cuda")
if args.dataset == 'mnist':
    z_fixed = z_fixed.squeeze()
x_fake_fixed = gen(z_fixed).detach()
print(x_fake_fixed.shape)
def criterion(predict, target, x_fake=x_fake_fixed):
    # fake_x = gen(z)
    predict_fake = model(x_fake)
    target_fake = torch.ones(size=(args.batch_size_real,)).to("cuda")
    return nn.BCELoss()(predict.squeeze(), target) + nn.BCELoss()(predict_fake.squeeze(),target_fake)

for img, label in dataloader:
    
    # predict =model(img.to("cuda").reshape(-1,784))
    predict = model(img.to("cuda"))
    target = torch.zeros(size=(args.batch_size_real,)).to("cuda")
    
    # sample_input = torch.rand(args.batch_size_fake, *img.shape[1:])
    # queried_model_output = model(sample_input)
    # if args.dataset == 'mnist':
    #     model_args = ("mlp",args.num_secure,784,1024,512,256,1)
    # elif args.dataset == 'cifar10':
    #     model_args = ("cnn",args.num_secure, (3,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0))
    # model_estimate_t1 = model_generator.get_model(*model_args)

    # recon_model()
    
    # print(predict.shape,target.shape)
    loss = criterion(predict, target)#nn.BCELoss()(predict.squeeze(),target)
    gradient = torch.autograd.grad(loss, model.parameters(),create_graph=True)
    gradient = [grad.detach() for grad in gradient]
    torchvision.utils.save_image(img,f"real_fl_{mtype}_{args.dataset}_{args.batch_size_real}.jpg")
    break
reconstruct_image = reconstruct_data(gradient, model, criterion, args)

torchvision.utils.save_image(reconstruct_image[:,:,:,:],f"test_fl_{mtype}_{args.dataset}_{args.batch_size_real}.jpg")
