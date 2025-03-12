from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import crypten
import crypten.communicator as comm
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
def train_hybrid_sim(args):
    # gen: public, torch model
    # disc: private part of discriminator (crypten model)
    # pub: public part of discriminator (torch model)
    # act: activation part of discriminator, public (torch model)
    # args: config argument

    # rank = comm.get().get_rank()
    gen = FC_Generator(args)
    disc_all = MLP_Disc(False,args.activation_type, args.num_secure, 784, 512,256,128,1)
    # disc_all = CNN_Disc(aargs.activation_type, args.num_secure, (3,16,3,2,1),(16,32,3,2,1),(32,64,3,2,1),(64,128,3,2,1))
    disc = disc_all.secured_layers
    pub = disc_all.public_layers
    # act = disc_all.activation
    # print(disc)
    gen.to(args.device)
    pub.to(args.device)
    # act.to(args.device)
    disc.to(args.device)
    disc_all.to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()#nn.BCEWithLogitsLoss()

    #load
    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen_{args.continue_from}.pth"
        pub_path = f"experiments/{args.save_path}/model/pub_{args.continue_from}.pth"
        act_path = f"experiments/{args.save_path}/model/act_{args.continue_from}.pth"
        disc_path = f"experiments/{args.save_path}/model/disc.encrypted.{args.continue_from}.pth"
        
        if os.path.exists(gen_path) and os.path.exists(pub_path) and os.path.exists(act_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            pub.load_state_dict(torch.load(pub_path))
            # act.load_state_dict(torch.load(act_path))
            disc.load_state_dict(torch.load(disc_path))
            
        else:
            raise FileNotFoundError

    #optimizer
    if args.optimizer == "sgd":
        momentum=0.5
        # g_opt = optim.Adam(gen.parameters(),lr=.0002,betas=[0.5,0.999])
        g_opt = optim.SGD(gen.parameters(), lr=args.learning_rate)#, momentum=0.4, nesterov=True)
        disc_private_opt = torch.optim.SGD(disc.parameters(), lr=args.learning_rate)#, momentum=momentum, nesterov=True)
        disc_public_opt = optim.SGD(pub.parameters(),lr=args.learning_rate)#, momentum=momentum, nesterov=True)
        disc_all_opt = optim.SGD(disc_all.parameters(),lr=args.learning_rate)#,momentum=momentum,nesterov=True)
        # disc_activation_opt = optim.SGD(act.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        g_opt = optim.Adam(gen.parameters(), lr=int(65536*args.learning_rate)/65536, betas=[0.5,0.999])
        disc_private_opt = torch.optim.Adam(disc.parameters(), lr=args.learning_rate,betas=[0.5,0.999])
        # disc_activation_opt = optim.Adam(act.parameters(),lr=args.learning_rate,betas=[0.5,0.999])
        disc_public_opt = optim.Adam(pub.parameters(), 
                                     lr=int(65536*args.learning_rate)/65536,betas=[0.5,0.999])
    
    # dataloader
    dataset= get_dataset(f"{args.dataset}_{args.model_type}", "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    iteration = args.continue_from
    accumulate_losses = {"gen":[], "disc":[]}
    fault = args.continue_from+1
    fixed_z = get_noise(args)
    reload = False
    pbar = tqdm(total = args.iteration_number)
    fa = open(f"experiments/{args.save_path}/train_log.csv","a+")
    # fa.write("Iteration,Generator Loss,Discriminator Loss\n")
    def get_gen_loss(criterion, output):
        target = torch.ones_like(output).to(args.device)
        return criterion(output,target)
    # calculate discriminator loss
    def get_disc_loss(criterion, real_output, fake_output):
        real_target = torch.ones_like(real_output)
        fake_target = torch.zeros_like(fake_output)
        return criterion(fake_output, fake_target) + criterion(real_output, real_target)

    while iteration < args.iteration_number:
        for img, _ in dataloader:
            # print(list(gen.parameters())[1])
            #==============generator update================
            if args.model_type == "fc":
                img = img.reshape(-1,np.prod(img.shape[1:]))
            # print(img.max())
            # assert False
            z = get_noise(args)
            fake_image = gen(z)
            if True:
                fake_image_enc = fake_image.detach()#crypten.cryptensor(fake_image).cpu()
                fake_image_enc.requires_grad=True
                fake_intermediate_output_enc = disc(fake_image_enc)
                fake_intermediate_output = fake_intermediate_output_enc.to(args.device)
                
                fake_intermediate_output_d = fake_intermediate_output.detach()
                fake_intermediate_output_d.requires_grad=True
                # fake_intermediate_output_d.zero_grad()
                fake_pub_output = pub(fake_intermediate_output_d)
                
                # fake_pub_output.requires_grad = True
                fake_pub_output = fake_pub_output.view(fake_pub_output.shape[0],-1)
                fake_output = fake_pub_output#act(fake_pub_output)
            
            else:
                fake_output = disc_all(fake_image)
            
            gen_loss = get_gen_loss(loss_fn, fake_output)
            
            # zero grad all models
            g_opt.zero_grad()
            disc_private_opt.zero_grad()
            disc_public_opt.zero_grad()
            if True:
                # backward calculation
                private_grad = torch.autograd.grad(gen_loss, fake_intermediate_output_d)[0]
                # print(private_grad.abs().mean())
                
                
                private_grad_enc = private_grad

                # backward pass such gradient
                fake_intermediate_output_enc.backward(private_grad_enc)
                # get gradient of the whole generator model
                public_gen_gradient = fake_image_enc.grad.to(args.device)
                # plt.hist(public_gen_gradient.cpu().numpy(),bins=50, alpha=0.7)
                # plt.savefig(f"public_gen_sim_{iteration}.jpg")
                print(public_gen_gradient.abs().mean())
                # backward pass the public gradient. Now the generator has all needed gradients
                fake_image.backward(public_gen_gradient)
            else:
                gen_loss.backward()
            g_opt.step()
            # np.savetxt("indirect_gen.txt",list(gen.parameters())[1].detach().numpy())
            # assert False
            #================discriminator update=========
            # get fake output
            if True:
                fake_image_detach            = fake_image.detach()
                fake_image_enc               = fake_image_detach#crypten.cryptensor(fake_image_detach).cpu()
                fake_image_enc.requires_grad = True
                fake_intermediate_output_enc = disc(fake_image_enc)
                fake_intermediate_output     = fake_intermediate_output_enc.to(args.device).detach()
                
                fake_intermediate_output.requires_grad = True
                fake_output                  = pub(fake_intermediate_output)
                fake_output                  = fake_output.view(fake_output.shape[0],-1)
                # fake_output                  = act(fake_output)

                # get real output
                real_image_enc               = img.to(args.device)#crypten.cryptensor(img)
                real_intermediate_output_enc = disc(real_image_enc)
                real_intermediate_output     = real_intermediate_output_enc.to(args.device).detach()
            
                real_intermediate_output.requires_grad = True
                real_output                  = pub(real_intermediate_output)
                real_output                  = real_output.view(real_output.shape[0],-1)
                # real_output                  = act(real_output)
            else:
                fake_output = disc_all(fake_image.detach())
                real_output = disc_all(img.to(args.device))
                
            disc_loss = get_disc_loss(loss_fn, real_output, fake_output)
            
            accumulate_losses['disc'].append(disc_loss)
            accumulate_losses['gen'].append(gen_loss.item())
            
            g_opt.zero_grad()
            if True:
                disc_private_opt.zero_grad()
                disc_public_opt.zero_grad()
            
                fake_grad = torch.autograd.grad(disc_loss, fake_intermediate_output,retain_graph=True)[0]
                real_grad = torch.autograd.grad(disc_loss, real_intermediate_output,retain_graph=True)[0]
                fake_grad_enc = fake_grad #crypten.cryptensor(fake_grad).cpu()
                real_grad_enc = real_grad #crypten.cryptensor(real_grad).cpu()

                # update public discriminator part gradient
                disc_loss.backward()
                
                # update private discriminator part gradient
                fake_intermediate_output_enc.backward(fake_grad_enc)
                real_intermediate_output_enc.backward(real_grad_enc)

                # update weight
                print(list(disc.parameters())[0].grad.abs().mean())
                disc_public_opt.step()
                disc_private_opt.step()
            else:
                disc_all_opt.zero_grad()
                disc_loss.backward()
                disc_all_opt.step()
            # np.savetxt("indirect_disc.txt",list(disc.parameters())[1].detach().numpy())
            # assert False
            # print(list(disc.parameters())[0].grad)
            # assert False
            pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
            if iteration % args.iteration_save == (args.iteration_save-1):
                # pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                fake_image = gen(fixed_z)
                ds = 0.5
                dm = 0.5
                fake_image = torch.clamp(fake_image * ds + dm, 0, 1)
                if args.model_type=="fc":
                    fake_image = fake_image.reshape(-1,1,28,28)
                torchvision.utils.save_image(fake_image, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration+1))
                fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                accumulate_losses = {"gen":[], "disc":[]} 
                
                # torch.save(gen.state_dict(),f"experiments/{args.save_path}/model/gen_{iteration+1}.pth")
                # torch.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.pth")
                # # crypten.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.{rank}")
                # torch.save(pub.state_dict(), f"experiments/{args.save_path}/model/pub_{iteration+1}.pth")
                # torch.save(act.state_dict(), f"experiments/{args.save_path}/model/act_{iteration+1}.pth")
            iteration += 1
            pbar.update(1)
            reload = False
            if iteration >= args.iteration_number:
                break
    pbar.close()
    
    


    

    
