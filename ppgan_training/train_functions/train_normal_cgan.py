from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
from tqdm import tqdm
import os
from torch.nn import Embedding
def train_normal(args):
    
    if args.model_type == "fc":
        if args.dataset.lower() == "mnist":
            # gen = MLP_Disc(True,args.activation_type, 0, args.zdim, 64, 128, 256, 512, 1024, 784)
            args.zdim += 10
            gen = FC_Generator(args)
            args.zdim -= 10
            disc= MLP_Disc(False,args.activation_type, 0, 794, 512, 256, 1)
        else:
            raise NotImplementedError
    elif args.model_type == "cnn":
        if args.dataset.lower() == "mnist":
            gen = DCGenerator(args)
            disc = CNN_Disc(args.activation_type, 0, (1,16,3,2,1),(16,32,3,2,1),(32,64,3,2,1),(64,128,3,2,1))#,(128,1,4,1,0))
        elif args.dataset.lower() == "celeba":
            gen = DCGenerator(args)
            disc = CNN_Disc(args.activation_type, 0, (3,16,3,2,1),(16,32,3,2,1),(32,64,3,2,1),(64,128,3,2,1))
    else:
        raise NotImplementedError
    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen_{args.continue_from}.pth"
        disc_path = f"experiments/{args.save_path}/model/disc_{args.continue_from}.pth"
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            disc.load_state_dict(torch.load(disc_path))
        else:
            raise FileNotFoundError
    # Check the device
    gen.to(args.device)
    disc.to(args.device)
    encoder = Embedding(10,10).to(args.device)
    torch.save(encoder,f"experiments/{args.save_path}/model/encoder.pth")
    # loss function and optimizer
    if args.optimizer == "sgd":
        momentum_G = args.momentum_G # 0.4
        momentum_D = args.momentum_D # 0.4
        
        d_opt = optim.SGD(disc.parameters(), lr=args.learning_rate,momentum=momentum_D, nesterov=True)#'''0.99'''
        g_opt = optim.SGD(gen.parameters(), lr=args.learning_rate,momentum=momentum_G, nesterov=True)#'''0.01'''
    elif args.optimizer=="adam":
        d_opt = optim.Adam(disc.parameters(), lr=args.learning_rate,betas=[0.5,0.999])
        g_opt = optim.Adam(gen.parameters(), lr=args.learning_rate,betas=[0.5,0.999])
    criterion = nn.BCELoss()
    def gan_loss(kind,*arguments):
        one = torch.ones_like(arguments[0],device=args.device)
        zero =torch.zeros_like(arguments[0],device=args.device)
        fake_output = arguments[0]
        if kind == "gen":
            return criterion(fake_output, one)
        elif kind == "disc":
            real_output = arguments[1]
            one = torch.ones_like(real_output,device=args.device)
            return criterion(real_output, one) + criterion(fake_output, zero)
        else:
            raise NotImplementedError
    # dataset
    dataset = get_dataset(f"{args.dataset}_{args.model_type}", "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    # data_iterator = iter(dataloader)
    iteration = args.continue_from#0
    accumulate_losses = {"gen":[], "disc":[]}
    pbar = tqdm(total = args.iteration_number)
    fixed_z = get_noise(args)
    z_lbl = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1]).int().to(args.device)
    fixed_z = torch.cat((encoder(z_lbl).to(args.device), fixed_z), -1)
    
    fa = open(f"experiments/{args.save_path}/train_log.csv","w+")
    fa.write("Iteration,Generator Loss,Discriminator Loss\n")
    while iteration < args.iteration_number:
        for img, lbl in dataloader:
            encoded_lbl = encoder(lbl.to(args.device))
            #fake image, update generator
            z = get_noise(args)
            z_lbl = torch.randint(10,size=(args.batch_size,))
            encoded_zlbl = encoder(z_lbl.to(args.device))
            z = torch.cat((encoded_zlbl, z), -1)
            fake_img = gen(z)
            fake_img = torch.cat((fake_img.view(fake_img.size(0), -1), encoded_zlbl), -1)
            fake_output = disc(fake_img)
            gen_loss = gan_loss("gen", fake_output)
            g_opt.zero_grad()
            gen_loss.backward()
            g_opt.step()
            accumulate_losses["gen"].append(gen_loss.item())
            # real image, update discriminator
            # print(img.shape)
            
            img = img.to(args.device)
            img = torch.cat((img.view(img.size(0), -1), encoded_lbl), -1)
            real_output = disc(img)
            z = get_noise(args)
            z_lbl = torch.randint(10,size=(args.batch_size,))
            encoded_zlbl = encoder(z_lbl.to(args.device))
            z = torch.cat((encoded_zlbl, z), -1)
            fake_img = gen(z)
            fake_img = torch.cat((fake_img.view(fake_img.size(0), -1), encoded_zlbl), -1)
            fake_output = disc(fake_img)

            disc_loss = gan_loss("disc", fake_output, real_output)
            d_opt.zero_grad()
            disc_loss.backward()
            d_opt.step()
            accumulate_losses["disc"].append(disc_loss.item())
            if iteration % args.iteration_save == (args.iteration_save-1):
                pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration+1,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                fake_img = gen(fixed_z)
                ds = 0.5
                dm = 0.5
                fake_img = torch.clamp(fake_img * ds + dm, 0, 1)
                if args.model_type=="fc":
                    fake_img = fake_img.reshape(-1,1,28,28)
                torchvision.utils.save_image(fake_img, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration+1))
                # with open(f"expereiments/{args.save_path}/train_log.txt","a") as fa:
                fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                accumulate_losses = {"gen":[], "disc":[]} 
            
                torch.save(gen.state_dict(), f"experiments/{args.save_path}/model/gen_{iteration+1}.pth")
                torch.save(disc.state_dict(), f"experiments/{args.save_path}/model/disc_{iteration+1}.pth")
            iteration += 1
            pbar.update(1)    
            if iteration >= args.iteration_number:
                break   
    pbar.close()