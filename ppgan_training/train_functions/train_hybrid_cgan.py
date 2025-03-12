from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import crypten
import crypten.communicator as comm
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from torch.nn import Embedding
def train_hybrid_cgan(gen, disc, pub, args):
    # gen: public, torch model
    # disc: private part of discriminator (crypten model)
    # pub: public part of discriminator (torch model)
    # act: activation part of discriminator, public (torch model)
    # args: config argument

    rank = comm.get().get_rank()

    gen.to(args.device)
    pub.to(args.device)
    # act.to(args.device)
    loss_fn = lambda x,y: (x-y).square().mean()#nn.BCEWithLogitsLoss()
    disc = disc.to(args.device)
    scaler = GradScaler()
    encoder = Embedding(10,10).to(args.device)
    torch.save(encoder,f"experiments/{args.save_path}/model/encoder.pth")
    # pub.half()
    # act.half()
    #load
    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen_{args.continue_from}.pth"
        pub_path = f"experiments/{args.save_path}/model/pub_{args.continue_from}.pth"
        # act_path = f"experiments/{args.save_path}/model/act_{args.continue_from}.pth"
        disc_path = f"experiments/{args.save_path}/model/disc.encrypted.{args.continue_from}.{rank}"
        
        if os.path.exists(gen_path) and os.path.exists(pub_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            pub.load_state_dict(torch.load(pub_path))
            # act.load_state_dict(torch.load(act_path))
            disc.load_state_dict(torch.load(disc_path))
            
        else:
            raise FileNotFoundError

    #optimizer
    if args.optimizer == "sgd":
        momentum = 0.4
        g_opt = optim.SGD(gen.parameters(), lr=args.learning_rate)#, momentum=momentum, nesterov=True)
        # g_opt = optim.Adam(gen.parameters(), lr=2e-4, betas=[0.5,0.999])
        disc_private_opt = crypten.optim.SGD(disc.parameters(), lr=args.learning_rate)#, momentum=momentum, nesterov=True)
        disc_public_opt = optim.SGD(pub.parameters(),lr=args.learning_rate)
        # disc_public_opt = optim.SGD(list(pub.parameters())+list(act.parameters()),lr=args.learning_rate)#, momentum=momentum, nesterov=True)
        # disc_activation_opt = optim.SGD(act.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        g_opt = optim.Adam(gen.parameters(), lr=args.learning_rate*2, betas=[0.5,0.999])
        disc_private_opt = crypten.optim.Adam(disc.parameters(), lr=args.learning_rate,betas=[0.5,0.999])
        # disc_activation_opt = optim.Adam(act.parameters(),lr=args.learning_rate,betas=[0.5,0.999])
        disc_public_opt = optim.Adam(pub.parameters(), 
                                     lr=args.learning_rate,betas=[0.5,0.999])
    
    # dataloader
    dataset= get_dataset(f"{args.dataset}_{args.model_type}", "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    
    iteration = args.continue_from
    accumulate_losses = {"gen":[], "disc":[]}
    fault = args.continue_from+1
    fixed_z = get_noise(args)
    z_lbl = torch.tensor([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1]).int().to('cuda')
    fixed_z = torch.cat((encoder(z_lbl).to(args.device), fixed_z), -1)
    reload = False
    pbar = tqdm(total = args.iteration_number)
    fa = open(f"experiments/{args.save_path}/train_log.csv","a+")
    # fa.write("Iteration,Generator Loss,Discriminator Loss\n")
    while iteration < args.iteration_number:
        for img, lbl in dataloader:
            # try:
                if img.shape[0] != args.batch_size:
                    continue
                # record previous model to reload in case of wrong secure computation execution
                # if not reload:
                #     try:
                #         last_last_disc = last_disc.copy()
                #         last_last_gen = last_gen.copy()
                #         last_last_pub = last_pub.copy()
                #     except:
                #         pass
                #     last_disc = disc.state_dict().copy()
                #     last_gen = gen.state_dict().copy()
                #     last_pub = pub.state_dict().copy()

                # label encoding
                encoded_lbl = encoder(lbl.to(args.device)).to(args.device)

                #==============generator update================
                if args.model_type == "fc":
                    img = img.reshape(-1,np.prod(img.shape[1:]))
                # print(img.max())
                # assert False
                z = get_noise(args)
                z_lbl = torch.randint(10,size=(args.batch_size,))
                encoded_zlbl = encoder(z_lbl.to(args.device))
                z = torch.cat((encoded_zlbl, z), -1)
                fake_image = gen(z)
                fake_image = torch.cat((fake_image.view(fake_image.size(0), -1), encoded_zlbl), -1)
                fake_image_enc = crypten.cryptensor(fake_image)#.cpu()
                # for p in disc.parameters():
                #     print(p.device)
                # print(fake_image_enc.device)
                fake_intermediate_output_enc = disc(fake_image_enc)
                fake_intermediate_output = fake_intermediate_output_enc.get_plain_text().to(args.device)
                if iteration % 10 == 9 or True:
                    fake_intermediate_output_d = fake_intermediate_output.detach()
                    fake_intermediate_output_d.requires_grad=True
                    # fake_intermediate_output_d.zero_grad()
                    with autocast(dtype=torch.float16):
                        fake_pub_output = pub(fake_intermediate_output_d)
                        
                        # fake_pub_output.requires_grad = True
                        fake_pub_output = fake_pub_output.view(fake_pub_output.shape[0],-1)
                        fake_output = fake_pub_output#act(fake_pub_output)
                    # print(fake_output)
                    def get_gen_loss(criterion, output):
                        target = torch.ones_like(output).to(args.device)
                        return criterion(output,target)
                    # with autocast(dtype=torch.float16):
                    gen_loss = get_gen_loss(loss_fn, fake_output)
                    if gen_loss.abs().max() > 1000:
                        print("Fall back due to overflow, gen", gen_loss.abs().max())
                        crypten._setup_prng(iteration+fault)
                        # gen.load_state_dict(last_last_gen)
                        # disc.load_state_dict(last_last_disc,strict=False)
                        # pub.load_state_dict(last_last_pub)
                        # last_gen = last_last_gen
                        # last_disc = last_last_disc
                        # last_pub = last_last_pub
                        fault += 1
                        reload=True
                        continue
                    # print(gen_loss.item())#.get_plain_text())
                    
                    # zero grad all models
                    # g_opt.zero_grad()
                    disc_private_opt.zero_grad()
                    disc_public_opt.zero_grad()
                    
                    # backward calculation

                    private_grad = torch.autograd.grad(gen_loss, fake_intermediate_output_d)[0]#.cpu()
                    
                        # assert False
                        # print(private_grad.abs().mean())
                    private_grad_enc = crypten.cryptensor(private_grad)
                    # print(private_grad_enc.get_plain_text().abs().mean())
                    # backward pass such gradient
                    fake_intermediate_output_enc.backward(grad_input=private_grad_enc, retain_graph=True)
                    # get gradient of the whole generator model
                    public_gen_gradient = fake_image_enc.grad.get_plain_text().to(args.device)
                    # print(public_gen_gradient.abs().mean())
                    # clear the gradient
                    # if rank == 1:   
                    #     plt.hist(public_gen_gradient.cpu().numpy(),bins=50, density=True, alpha=0.7)
                    #     plt.savefig(f"public_gen_hybrid_{iteration}.jpg")
                    # backward pass the public gradient. Now the generator has all needed gradients
                    fake_image.backward(public_gen_gradient)
                    # print(list(gen.parameters())[0].grad)
                    g_opt.step()

                    accumulate_losses['gen'].append(gen_loss.detach().item())
                
                #================discriminator update=========
                # get fake output
                g_opt.zero_grad()
                disc_private_opt.zero_grad()
                disc_public_opt.zero_grad()
                # z = get_noise(args)
                # fake_image = gen(z)
                # fake_image_detach            = fake_image.detach()
                # fake_image_enc               = crypten.cryptensor(fake_image_detach)#.cpu()
                # fake_intermediate_output_enc = disc(fake_image_enc)
                fake_intermediate_output     = fake_intermediate_output_enc.get_plain_text().to(args.device)
                with autocast(dtype=torch.float16):
                    fake_intermediate_output.requires_grad = True
                    fake_output                  = pub(fake_intermediate_output)
                    fake_output                  = fake_output.view(fake_output.shape[0],-1)
                    # fake_output                  = act(fake_output)

                # get real output
                img = img.to(args.device)
                img = torch.cat((img.view(img.size(0), -1), encoded_lbl), -1)
                real_image_enc               = crypten.cryptensor(img).to(args.device)
                real_intermediate_output_enc = disc(real_image_enc)
                real_intermediate_output     = real_intermediate_output_enc.get_plain_text().to(args.device)
                # if fake_intermediate_output.abs().max() > 1000:
                #     print("Fall back due to overflow, disc real")
                #     crypten._setup_prng(iteration+fault)
                #     gen.load_state_dict(last_last_gen)
                #     disc.load_state_dict(last_last_disc,strict=False)
                #     pub.load_state_dict(last_last_pub)
                #     # last_gen = last_last_gen
                #     # last_disc = last_last_disc
                #     # last_pub = last_last_pub
                #     fault += 1
                #     reload = True
                #     continue

                real_intermediate_output.requires_grad = True
                with autocast(dtype=torch.float16):
                    real_output                  = pub(real_intermediate_output)
                    real_output                  = real_output.view(real_output.shape[0],-1)
                    # real_output                  = act(real_output)

                # calculate discriminator loss
                def get_disc_loss(criterion, real_output, fake_output):
                    real_target = torch.ones_like(real_output)
                    fake_target = torch.zeros_like(fake_output)
                    return criterion(fake_output, fake_target) + criterion(real_output, real_target)
                with autocast(dtype=torch.float16):
                    disc_loss = get_disc_loss(loss_fn, real_output, fake_output)
                # if disc_loss < 0.25: continue
                if disc_loss.abs().max() > 1000:
                    print("Fall back due to overflow, disc fake")
                    crypten._setup_prng(iteration+fault)
                    # gen.load_state_dict(last_last_gen)
                    # disc.load_state_dict(last_last_disc,strict=False)
                    # pub.load_state_dict(last_last_pub)
                    # last_gen = last_last_gen
                    # last_disc = last_last_disc
                    # last_pub = last_last_pub
                    fault += 1
                    reload = True
                    continue
                accumulate_losses['disc'].append(disc_loss.detach())
                
                
                fake_grad = torch.autograd.grad(disc_loss, fake_intermediate_output,retain_graph=True)[0]
                real_grad = torch.autograd.grad(disc_loss, real_intermediate_output,retain_graph=True)[0]
                fake_grad_enc = crypten.cryptensor(fake_grad)#.cpu()
                real_grad_enc = crypten.cryptensor(real_grad)#.cpu()

                # update public discriminator part gradient
                scaler.scale(disc_loss).backward()
                
                # update private discriminator part gradient
                fake_intermediate_output_enc.backward(fake_grad_enc)
                real_intermediate_output_enc.backward(real_grad_enc)
                # fake_intermediate_output_enc.backward()
                
                # for p in pub.parameters():
                #     if p.grad is not None:
                #         p.grad = ((p.grad*2**16).int()).float()/2**16
                # print(list(disc.parameters())[0].grad.get_plain_text())
                # real_intermediate_output_enc.backward(real_grad_enc)
                # update weight
                print(list(disc.parameters())[0].grad.get_plain_text().abs().mean())
                # disc_public_opt.step()
                disc_private_opt.step()
                scaler.step(disc_public_opt)
                scaler.update()
                # fake_intermediate_output_enc.clean_graph()
                # assert False
               
                if iteration % 10 == 9 or True:
                    pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                if iteration % args.iteration_save == (args.iteration_save-1):
                    
                    # pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                    if rank==1:
                        with torch.no_grad():
                            fake_image = gen(fixed_z).detach()
                        ds = 0.5
                        dm = 0.5
                        fake_image = torch.clamp(fake_image * ds + dm, 0, 1)
                        if args.model_type=="fc":
                            fake_image = fake_image.reshape(-1,1,28,28)
                        torchvision.utils.save_image(fake_image, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration+1))
                        fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                    accumulate_losses = {"gen":[], "disc":[]} 
                    if iteration % 5000 == 4999:
                        torch.save(gen.state_dict(),f"experiments/{args.save_path}/model/gen_{iteration+1}.pth")
                        crypten.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.{rank}")
                        torch.save(pub.state_dict(), f"experiments/{args.save_path}/model/pub_{iteration+1}.pth")
                        # torch.save(act.state_dict(), f"experiments/{args.save_path}/model/act_{iteration+1}.pth")
                iteration += 1
                pbar.update(1)
                reload = False
                if iteration >= args.iteration_number:
                    break
            # except:
                # continue
    pbar.close()
    
    


    

    
