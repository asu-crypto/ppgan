from random import seed
import torch
from torchvision.utils import save_image
from evaluation.inception import inception_score
from util import seed_everything
from torch.nn import Embedding
# seed_everything(1234)
def get_noise(args):
    # if args.model_type == "cnn":
        # return torch.randn(args.batch_size, args.zdim, 1, 1).to(args.device)
    # elif args.model_type == "fc":
    args.batch_size=4
    return torch.randn(args.batch_size, args.zdim).to(args.device)

from torch.utils.data import Dataset
class OutputDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self,idx):
        return self.imgs[idx,:,:,:]

def generate_images(model, args):
    model.eval()
    outputs = []
    z = get_noise(args)
    
    # print(label.shape)
    encoder = torch.load(f"experiments/exp_type_4_0/model/encoder.pth")#Embedding(10,10).to(args.device)
    outputs = []
    for i in range(10):
        z = get_noise(args)
        label = torch.tensor([i]*args.batch_size,device=args.device).int()
        # print(z.shape)
        z = torch.cat((encoder(label).to(args.device), z), -1)
        output = model(z)
        ds = 0.5
        dm = 0.5
        output = torch.clamp(output * ds + dm, 0, 1)
        output = output.reshape(-1,1,28,28)
        outputs.append(output)
    output = torch.cat(outputs,dim=0)
    save_image(output[:,:,:,:],f"tmp_generated/test_{i}.jpg",nrow=args.batch_size)
        # output = output.reshape(-1,1,28,28)
        # outputs.append(output)

    # output = torch.cat(outputs, dim=0)
    # print(output.shape)
    # output_dataset = OutputDataset(output)
    # print(len(output_dataset))
    # for i in range(output.shape[0]):
    #     save_image(output[i,:,:,:],f"tmp_generated/test_{i}.jpg")
    # iscore = inception_score.inception_score(output_dataset,resize=True)
    # print(iscore)



from model import *
from options import options
args = options().parse_args()
args.zdim += 10
gen = FC_Generator(args)#DCGenerator(args).to(args.device)#MLP_Disc(True,args.activation_type, 0, args.zdim, 128, 256, 512, 1024, 784)#FC_Generator(args)#
args.zdim -= 10
gen.to(args.device)
i = int(args.continue_from)
gen.load_state_dict(torch.load(f"experiments/exp_type_4_0/model/gen_{i*100}.pth"))
generate_images(gen, args)
# for i in range(1,51):
# # # i=30
#     gen.load_state_dict(torch.load(f"experiments/exp_type_3_5/model/gen_{i*1000}.pth"))
print(i*200, end=",")
#     generate_images(gen, args)