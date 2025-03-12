from options import *
from model import *
from util import *
from dataset import get_dataset
from torch.utils.data import DataLoader
from reconstruct_model import recon_model

args=options().parse_args()
seed_everything(args.seed)
model_generator = ModelGenerator()

disc = model_generator.get_model("mlp",args.num_secure,784,1024,512,256,1).to("cuda")
# disc = model_generator.get_model("cnn",args.num_secure, (1,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0)).to("cuda")
print("Number of parameters to reconstruct:", weight_count(disc.secured_layers))

dataset = get_dataset(args.dataset,"train")
dataloader = DataLoader(dataset, batch_size=args.batch_size_real)
 
def l2loss(a,b):
    return ((a-b).square().mean()).sqrt()



# weight recovery attack
sample_input = torch.eye(784).to("cuda")#torch.rand(args.batch_size_fake, 1, 784).to("cuda")
sample_output = disc.secured_layers(sample_input)
print(sample_input.shape, sample_output.shape)
sample_zero = torch.zeros(1,784).to("cuda")
sample_output1 = disc.secured_layers(sample_zero)
sample_output -= sample_output1
# print(model.)
# model_estimate = recon_model(sample_input, sample_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, "mlp",args.num_secure,784,1024,512,256,1)
# model_estimate = recon_model(sample_input, sample_output.detach().clone(), args.num_recon_iteration, 1000, l2loss, model_generator, "cnn",args.num_secure, (1,128,4,2,1),(128,256,4,2,1),(256,512,4,2,1),(512,1024,4,2,1),(1024,1,4,1,0))

# estimated = list(model_estimate.secured_layers.parameters())
actual = list(disc.secured_layers.parameters())
print(actual[0])
print(sample_output.T)
print(actual[1])
print(sample_output1)
assert False
print("BENCHMARK DISTANCE BETWEEN MODEL AND ITS RECOVERY:")
print("Similarity Distance:",sim_dist(estimated, actual))
print("MAE Distance:",mae_dist(estimated, actual))
print("MSE Distance:",mse_dist(estimated, actual))

# # 