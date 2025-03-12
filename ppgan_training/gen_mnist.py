from random import seed
import torch
from torchvision.utils import save_image
from evaluation.inception import inception_score
from util import seed_everything
from torch.nn import Embedding
from tqdm import tqdm

seed_everything(1234)

def get_noise(args):
    # if args.model_type == "cnn":
        # return torch.randn(args.batch_size, args.zdim, 1, 1).to(args.device)
    # elif args.model_type == "fc":
    args.batch_size=1
    return torch.randn(args.batch_size, args.zdim).to(args.device)

def generate_images(model, args):
    model.eval()
    outputs = []
    z = get_noise(args)
    
    # print(label.shape)
    encoder = torch.load(f"experiments/exp_type_4_2/model/encoder.pth")#Embedding(10,10).to(args.device)
    outputs = []
    n_class = np.zeros(10)
    n_class[0] = 5923
    n_class[1] = 6742
    n_class[2] = 5958
    n_class[3] = 6131
    n_class[4] = 5842
    n_class[5] = 5421
    n_class[6] = 5918
    n_class[7] = 6265
    n_class[8] = 5851
    n_class[9] = 5949
    n_image = int(sum(n_class))
    image_lables = np.zeros(shape=[n_image,])#, len(n_class)])

    image_cntr = 0
    for class_cntr in np.arange(len(n_class)):
        for cntr in np.arange(n_class[class_cntr]):
            image_lables[image_cntr] = class_cntr#, class_cntr] = 1
            image_cntr += 1
    for i in tqdm(range(n_image)):
        z = get_noise(args)
        label = torch.tensor([image_lables[i],],device=args.device).int()
        # print(z.shape)
        z = torch.cat((encoder(label).to(args.device), z), -1)
        output = model(z)
        ds = 0.5
        dm = 0.5
        output = torch.clamp(output * ds + dm, 0, 1)
        output = output.reshape(-1,1,28,28)
        outputs.append(output)
    output = torch.cat(outputs,dim=0)
    torch.save(output, "cgan-mnist-hybrid-images.pth")
    torch.save(torch.from_numpy(image_lables),"cgan-mnist-hybrid-labels.pth")
    # save_image(output[:,:,:,:],f"tmp_generated/test_{i}.jpg",nrow=args.batch_size)
       



from model import *
from options import options
args = options().parse_args()
args.zdim += 10
gen = FC_Generator(args)#DCGenerator(args).to(args.device)#MLP_Disc(True,args.activation_type, 0, args.zdim, 128, 256, 512, 1024, 784)#FC_Generator(args)#
args.zdim -= 10
gen.to(args.device)
i = int(args.continue_from)
gen.load_state_dict(torch.load(f"experiments/exp_type_4_2/model/gen_{i*100}.pth"))
generate_images(gen, args)
# for i in range(1,51):
# # # i=30
#     gen.load_state_dict(torch.load(f"experiments/exp_type_3_5/model/gen_{i*1000}.pth"))
# print(i*200, end=",")
#     generate_images(gen, args)