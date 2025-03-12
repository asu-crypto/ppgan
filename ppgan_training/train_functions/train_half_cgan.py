# from tkinter import E
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import crypten
import torch
import crypten.communicator as comm
from tqdm import tqdm
import os 
from torch.nn import Embedding

def train_half_cgan(gen, disc, args):
    # train half secure gan (secure disc, normal gen)
    rank = comm.get().get_rank()
    gen.to(args.device)
    disc.to(args.device)
    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen_{args.continue_from}.pth"
        disc_path = f"experiments/{args.save_path}/model/disc.encrypted.{args.continue_from}.{rank}"
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            disc.load_state_dict(torch.load(disc_path))
        else:
            raise FileNotFoundError
    # disc.to(args.device), no gpu since gpu seems slower
    def get_gen_loss(criterion, output):
        fake_target = crypten.cryptensor(torch.zeros(*output.shape)+1).to(args.device)
        return criterion(output,fake_target)
    def get_disc_loss(criterion, fake_output, real_output):
        fake_target = crypten.cryptensor(torch.zeros(*fake_output.shape)).to(args.device)
        real_target = crypten.cryptensor(torch.zeros(*real_output.shape)+1).to(args.device)
        return criterion(fake_output,fake_target) + criterion(real_output,real_target)
    # optimizer
    if args.optimizer == "sgd":
        g_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=[0.5,0.999])#optim.SGD(gen.parameters(), lr=args.learning_rate,momentum=0.4,nesterov=True)
        d_opt = crypten.optim.SGD(disc.parameters(), lr=args.learning_rate,momentum=0.4,nesterov=True)
    elif args.optimizer == "adam":
        g_opt = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=[0.5, 0.999])#,grad_threshold=0.5)
        d_opt = crypten.optim.Adam(disc.parameters(), lr=args.learning_rate, betas=[0.5, 0.999],grad_threshold=0.5)
    bceloss = lambda x,y: (x-y).square().mean()#crypten.nn.BCELoss()
    dataset= get_dataset(f"{args.dataset}_{args.model_type}", "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    iteration = args.continue_from+1#0
    accumulate_losses = {"gen":[], "disc":[]}
    fault =0
    reload=False
    encoder = Embedding(10,10).to(args.device)
    torch.save(encoder,f"experiments/{args.save_path}/model/encoder.pth")
    # rereload = False
    reload_iter = -100
    pbar = tqdm(total = args.iteration_number)
    fa = open(f"experiments/{args.save_path}/train_log.csv","a+")
    fa.write("Iteration,Generator Loss,Discriminator Loss\n")
    fixed_z = get_noise(args)
    z_lbl = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1]).int().to(args.device)
    fixed_z = torch.cat((encoder(z_lbl).to(args.device), fixed_z), -1)
    
    while iteration < args.iteration_number:
        for img, lbl in dataloader:
            if img.shape[0] < args.batch_size: continue
            if args.model_type=="fc":
                img = img.reshape(-1,np.prod(img.shape[1:]))
            encoded_lbl = encoder(lbl.to(args.device))
            # generator update
            # noise sample
            z = get_noise(args)
            z_lbl = torch.randint(10,size=(args.batch_size,))
            encoded_zlbl = encoder(z_lbl.to(args.device))
            z = torch.cat((encoded_zlbl, z), -1)
            
            fake_img = gen(z) 
            fake_img = torch.cat((fake_img.view(fake_img.size(0), -1), encoded_zlbl), -1)
            fake_img_enc = crypten.cryptensor(fake_img)
            fake_img_enc.require_grad=True
            fake_output = disc(fake_img_enc)#.cpu())
            
            gen_loss = get_gen_loss(bceloss, fake_output)
            if(gen_loss.get_plain_text().abs().max() > 100):
                fault+=1
                print("gen error", fake_output.get_plain_text().abs().max(), fake_img_enc.get_plain_text().abs().max())
                for p in disc.parameters():
                    print(p.get_plain_text().abs().max())
                # print(list(disc.parameters()).get_plain_text().abs().max())
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen)
                disc.load_state_dict(last_last_disc,strict=False)
                # print(list(disc.parameters())[2].get_plain_text().abs().max())
                # fault += 1
                reload_iter = iteration
                reload=True
                continue
            disc.zero_grad()
            g_opt.zero_grad()
            gen_loss.backward()#retain_graph=True)
            grad = fake_img_enc.grad.get_plain_text().to(args.device)
            # mask = torch.rand(grad.shape).to(args.device)
            # grad -= mask
            # print(grad.abs().mean())
            g_opt.zero_grad()
            fake_img.backward(grad)#,retain_graph=True)
            
            
            # fake_img.backward(mask)
            # print(grad)
            # print(list(gen.parameters())[0].grad)
            g_opt.step()
            # print(gen_loss.get_plain_text())
            
            # else: fault = 0
            # def update_disc(fault, rerun = False):
                # global gen, disc
                # discriminator update
                # noise sample
                
            fake_img_enc = crypten.cryptensor(fake_img.detach())#.cpu()
            fake_output = disc(fake_img_enc)
            # real image
            img = img.to(args.device)
            img = torch.cat((img.view(img.size(0), -1), encoded_lbl), -1)
            crypt_img = crypten.cryptensor(img).to(args.device) 
            real_output = disc(crypt_img)
            disc_loss = get_disc_loss(bceloss, fake_output,real_output)
            # return disc_loss
        
            if(disc_loss.get_plain_text().abs().max() > 100):
                fault+=1
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen)
                disc.load_state_dict(last_last_disc,strict=False)
                # fault += 1
                reload=True
                print("disc error", real_output.get_plain_text().abs().max(), crypt_img.get_plain_text().abs().max())
                continue
            # return update_disc(fault+1, True)
            # return  gen, disc, disc_loss, rerun, fault
            # gen, disc, disc_loss, rerun, fault = update_disc(fault)
            # reload = rerun
            # if rerun:
            #     last_disc = last_last_disc
            #     last_gen = last_last_gen
            #     print("reruned")
            # else: fault=0
            d_opt.zero_grad()
            
            if not reload:
                try:
                    last_last_gen = last_gen.copy()
                    last_last_disc = last_disc.copy()
                except:
                    pass
                last_gen = gen.state_dict().copy()
                last_disc = disc.state_dict().copy()
            disc_loss.backward()
            d_opt.step()
            accumulate_losses['disc'].append(disc_loss.get_plain_text())
            accumulate_losses['gen'].append(gen_loss.get_plain_text())
            pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
            if False:
                print(crypten.multiplication_time)
                crypten.multiplication_time = 0
                try:
                    print("Num compare:",crypten.num_compare)
                    crypten.num_compare = 0
                except:
                    print("Num compare: 0")
            if iteration % args.iteration_save == (args.iteration_save-1):
                # pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration+1,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                fake_img = gen(fixed_z).detach()
                if args.model_type=="fc":
                    fake_img = fake_img.reshape(-1,1,28,28)
                torchvision.utils.save_image(fake_img, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration+1))
                fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                accumulate_losses = {"gen":[], "disc":[]} 
                torch.save(gen.state_dict(),f"experiments/{args.save_path}/model/gen_{iteration+1}.pth")
                crypten.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.{rank}")
            iteration += 1
            pbar.update(1)
            # reload = rereload
            if iteration > reload_iter + 1:
                reload = False
            if iteration == args.iteration_number:
                break
    pbar.close()
