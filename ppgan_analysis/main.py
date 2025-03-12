from typing import Counter
from options import *
from model import *
from reconstruct_data import reconstruct_data
from util import *
from dataset import get_dataset
from torch.utils.data import DataLoader
from reconstruct_model import recon_model
import torchvision
import time
from metrics import *
args=options().parse_args()
seed_everything(args.seed)
model_generator = ModelGenerator()

mtype="fully_connected" if args.dataset=="mnist" else "deep_convolution"
if args.dataset == "mnist":
    model = model_generator.get_model("mlp",args.num_secure,784,1024,512,256,1).to("cuda")
    model_clone = model_generator.get_model("mlp",args.num_secure,784,1024,512,256,1).to("cuda")
    gen = model_generator.get_model("mlp",0, 100,256,512,1024,784).to("cuda")
elif args.dataset == "cifar10" or args.dataset == "celeba":
    model = model_generator.get_model("cnn", args.num_secure, (3,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
    model_clone = model_generator.get_model("cnn", args.num_secure, (3,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
    gen = DCGenerator().to("cuda")
# disc = model_generator.get_model("cnn",args.num_secure, (1,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
# model_clone = model.clone()
model_clone.load_state_dict(model.state_dict())
if args.num_secure > 0:
    print("Number of parameters to reconstruct:", weight_count(model.secured_layers))

dataset = get_dataset(args.dataset,"train")
dataloader = DataLoader(dataset, batch_size=args.batch_size_real,shuffle=False)
# print(len(dataset))
def l2loss(a,b):
    return ((a-b).square().mean()).sqrt()


optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

z_fixed = torch.randn(args.batch_size_real, 100, 1, 1).to("cuda")
if args.dataset == 'mnist':
    z_fixed = z_fixed.reshape(args.batch_size_real, 100)
x_fake_fixed = gen(z_fixed).detach()
# print(x_fake_fixed.shape)
def criterion(predict, target, x_fake=x_fake_fixed):
    # fake_x = gen(z)
    # print("called")
    predict_fake = model_clone(x_fake)
    target_fake = torch.ones(size=(args.batch_size_real,)).to("cuda")
    return nn.BCELoss()(predict.reshape(-1), target) + nn.BCELoss()(predict_fake.reshape(-1),target_fake)
it = 0
t = time.time()
import copy
model_clone_2 = copy.deepcopy(model_clone)
for img, label in dataloader:
    if it < 20:
        model = copy.deepcopy(model_clone_2)
        torchvision.utils.save_image(img,f"exp/real_fl_{mtype}_{args.dataset}_{args.batch_size_real}_{it}.jpg")
        real_img = img
        # predict =model(img.to("cuda").reshape(-1,784))
        predict = model(img.to("cuda"))

        target = torch.zeros(size=(args.batch_size_real,)).to("cuda")
        
        sample_input = torch.rand(args.batch_size_fake, *img.shape[1:]).to("cuda")
        if args.dataset == "mnist":
            sample_input = sample_input.reshape(-1,np.prod(sample_input.shape[1:]))
        print(sample_input.shape)
        queried_model_output = model.secured_layers(sample_input)

        if args.num_secure > 0:
            if args.dataset == 'mnist':
                model_args = ("mlp",args.num_secure,784,1024,512,256,1)
            elif args.dataset == 'cifar10' or args.dataset=="celeba":
                model_args = ("cnn",args.num_secure, (3,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0))
        
            model_estimate_t1 = recon_model(sample_input, queried_model_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, *model_args)
            param1 = [p.detach() for p in model_estimate_t1.parameters()]
            estimated = list(model_estimate_t1.parameters())
            actual = list(model.secured_layers.parameters())
            print("BENCHMARK DISTANCE BETWEEN MODEL AND ITS RECOVERY:")
            print("Similarity Distance:",sim_dist(estimated, actual))
            print("MAE Distance:",mae_dist(estimated, actual))
            print("MSE Distance:",mse_dist(estimated, actual))
            # param2 = [p.detach() for p in model_estimate_t1.parameters()]
            # private_gradient = [(p-q)/0.1 for p,q in zip(param1,param2)]
            # predict = model_estimate_t1()
        
        # print(predict.shape,target.shape)
        loss = criterion(predict, target)#nn.BCELoss()(predict.squeeze(),target)
        
        
        if args.num_secure > 0:
            public_gradient = torch.autograd.grad(loss, model.public_layers.parameters(),create_graph=True)
            public_gradient = [grad.detach() for grad in public_gradient]
            true_gradient = [grad.detach() for grad in torch.autograd.grad(loss, model.parameters(), create_graph=True)]
            loss.backward()
            optimizer.step()
            queried_updated_model_output = model.secured_layers(sample_input)
            model_estimate_t2 = recon_model(sample_input, queried_model_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, *model_args)
            estimated = list(model_estimate_t2.parameters())
            actual = list(model.secured_layers.parameters())
            print("BENCHMARK DISTANCE BETWEEN MODEL AND ITS RECOVERY:")
            print("Similarity Distance:",sim_dist(estimated, actual))
            print("MAE Distance:",mae_dist(estimated, actual))
            print("MSE Distance:",mse_dist(estimated, actual))
            param2 = [p.detach() for p in model_estimate_t2.parameters()]
            private_gradient = [(p-q)/0.1 for p,q in zip(param1,param2)]
            

            
            input_gradient = private_gradient
            input_gradient.extend(public_gradient)
            print((input_gradient[0]-true_gradient[0]).abs().max())
            print(len(input_gradient))
            reconstruct_image = reconstruct_data(input_gradient, model_clone, criterion, args)
        else:
            input_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            input_gradient = [grad.detach() for grad in input_gradient] 
            print(len(input_gradient))
            reconstruct_image = reconstruct_data(input_gradient, model, criterion, args)
        torchvision.utils.save_image(reconstruct_image[:,:,:,:],f"exp/inverted_{mtype}_{args.num_secure}_secure_{args.dataset}_bsr_{args.batch_size_real}_bsf_{args.batch_size_fake}_{it}.jpg")

        ds = torch.as_tensor([0.5,0.5,0.5],device='cuda')[:,None,None]
        dm = torch.as_tensor([0.5,0.5,0.5],device='cuda')[:,None,None]
        rec_denormalized = torch.clamp(reconstruct_image * ds + dm, 0, 1)
        ground_truth_denormalized = torch.clamp(real_img.to('cuda') * ds + dm, 0, 1)

        print("PSNR:",psnr_compute(rec_denormalized,ground_truth_denormalized))
        print("CW SSIM:",cw_ssim(rec_denormalized,ground_truth_denormalized))
        if args.num_secure==0:
            print("FMSE:",feat_mse(rec_denormalized,real_img,model))
        else:
            print("FMSE:",feat_mse(rec_denormalized,real_img,model_clone))
        it += 1
    else:
        break

#
print(time.time()-t)