from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import crypten
import crypten.communicator as comm
from tqdm import tqdm
import os

def train_full(gen, disc, args):
    # args.device = "cpu"
    rank = comm.get().get_rank()
    gen.to(args.device)
    disc.to(args.device) #, no gpu since gpu seems slower

    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen.encrypted.{args.continue_from}.{rank}"
        disc_path = f"experiments/{args.save_path}/model/disc.encrypted.{args.continue_from}.{rank}"
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            disc.load_state_dict(torch.load(disc_path))
        else:
            raise FileNotFoundError

    def get_gen_loss(criterion, output):
        fake_target = crypten.cryptensor(torch.zeros(*output.shape)+1).to(args.device)
        return criterion(output.sigmoid(),fake_target)
    def get_disc_loss(criterion, fake_output, real_output):
        fake_target = crypten.cryptensor(torch.zeros(*fake_output.shape)).to(args.device)
        real_target = crypten.cryptensor(torch.zeros(*real_output.shape)+1).to(args.device)
        return criterion(fake_output.sigmoid(),fake_target) + criterion(real_output.sigmoid(),real_target)
    # optimizer
    if args.optimizer == "sgd":
        g_opt = crypten.optim.SGD(gen.parameters(), lr=args.learning_rate, momentum=0.1, nesterov=True)
        d_opt = crypten.optim.SGD(disc.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    elif args.optimizer == "adam":
        g_opt = crypten.optim.Adam(gen.parameters(), lr=args.learning_rate, betas=[0.5, 0.999])
        d_opt = crypten.optim.Adam(disc.parameters(), lr=args.learning_rate, betas=[0.5, 0.999])
    bceloss = crypten.nn.BCELoss()
    dataset= get_dataset(f"{args.dataset}_{args.model_type}", "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    iteration = args.continue_from#0
    accumulate_losses = {"gen":[], "disc":[]}
    fault =0
    reload = False
    pbar = tqdm(total = args.iteration_number)
    fa = open(f"experiments/{args.save_path}/train_log.csv","a+")
    fa.write("Iteration,Generator Loss,Discriminator Loss\n")
    fixed_z = get_noise(args)
    fixed_z_enc = crypten.cryptensor(fixed_z)
    while iteration < args.iteration_number:
        for img, _ in dataloader:
            # record previous model to reload in case of wrong secure computation execution
            if img.shape[0] < args.batch_size: continue
            if args.model == "fc":
                img = img.reshape(img.shape[0],-1)
            if not reload:
                try:
                    last_last_disc = last_disc.copy()
                    last_last_gen = last_gen.copy()
                    if args.optimizer=='adam':
                        last_last_state_gen = last_state_gen.copy()
                        last_last_state_disc = last_state_disc.copy()
                except Exception as e:
                    print(e)
                    # pass
                last_disc = disc.state_dict().copy()
                last_gen = gen.state_dict().copy()
                if args.optimizer=='adam':
                    last_state_gen = g_opt.state.copy()
                    last_state_disc = d_opt.state.copy()
            # generator update
            z = get_noise(args)
            z_enc = crypten.cryptensor(z)
            fake_image = gen(z_enc)
            if fake_image.get_plain_text().abs().max() > 1000:
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen, strict=False)
                disc.load_state_dict(last_last_disc,strict=False)
                fault += 1
                reload=True
                continue
            fake_output = disc(fake_image)
            gen_loss = get_gen_loss(bceloss, fake_output)
            if gen_loss.get_plain_text().abs().max() > 1000:
                print("gen error")
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen, strict = False)
                disc.load_state_dict(last_last_disc,strict=False)
                if args.optimizer == 'adam':
                    g_opt.state = last_last_state_gen
                    d_opt.state = last_last_state_disc
                fault += 1
                reload=True
                continue
            g_opt.zero_grad()
            gen_loss.backward()
            g_opt.step()
            
            # discriminator update
            fake_image_detach = fake_image.detach()
            fake_output = disc(fake_image_detach)
            
            real_image_crypt = crypten.cryptensor(img).to(args.device)
            real_output = disc(real_image_crypt)
            
            disc_loss = get_disc_loss(bceloss, fake_output, real_output)
            if disc_loss.get_plain_text().abs().max() > 1000:
                print("disc error")
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen, strict = False)
                disc.load_state_dict(last_last_disc,strict=False)
                if args.optimizer == 'adam':
                    g_opt.state = last_last_state_gen
                    d_opt.state = last_last_state_disc
                fault += 1
                reload=True
                continue
            d_opt.zero_grad()
            disc_loss.backward()
            d_opt.step()
            reload = False
            
            accumulate_losses['gen'].append(gen_loss.get_plain_text())
            accumulate_losses['disc'].append(disc_loss.get_plain_text())
            pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration+1,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
            if iteration % args.iteration_save == (args.iteration_save-1):
                
                fake_image_detach = gen(fixed_z_enc).detach()
                if args.model_type=="fc":
                    fake_image_detach = fake_image_detach.get_plain_text().reshape(-1,1,28,28)
                torchvision.utils.save_image(fake_image_detach, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration+1))
                fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                accumulate_losses = {"gen":[], "disc":[]} 
                crypten.save(gen.state_dict(),f"experiments/{args.save_path}/model/gen.encrypted.{iteration+1}.{rank}")
                crypten.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.{rank}")

            iteration += 1
            pbar.update(1)
            if iteration >= args.iteration_number:
                break
    pbar.close()


    