from model import *
from dataset import *
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import crypten
import crypten.communicator as comm
from tqdm import tqdm
import os

def train_hybrid_wgan(gen, disc, pub, act, args):
    # gen: public, torch model
    # disc: private part of discriminator (crypten model)
    # pub: public part of discriminator (torch model)
    # act: activation part of discriminator, public (torch model)
    # args: config argument

    rank = comm.get().get_rank()

    gen.to(args.device)
    pub.to(args.device)
    act.to(args.device)
    # loss_fn = torch.mean()

    #load
    if args.continue_from != -1:
        gen_path = f"experiments/{args.save_path}/model/gen_{args.continue_from}.pth"
        pub_path = f"experiments/{args.save_path}/model/pub_{args.continue_from}.pth"
        act_path = f"experiments/{args.save_path}/model/act_{args.continue_from}.pth"
        disc_path = f"experiments/{args.save_path}/model/disc.encrypted.{args.continue_from}.{rank}"
        
        if os.path.exists(gen_path) and os.path.exists(pub_path) and os.path.exists(act_path) and os.path.exists(disc_path):
            gen.load_state_dict(torch.load(gen_path))
            pub.load_state_dict(torch.load(pub_path))
            act.load_state_dict(torch.load(act_path))
            disc.load_state_dict(torch.load(disc_path))
            
        else:
            raise FileNotFoundError

    #optimizer
    if args.optimizer == "sgd":
        momentum = 0.5
        g_opt = optim.SGD(gen.parameters(), lr=args.learning_rate, momentum=0.4, nesterov=True)
        disc_private_opt = crypten.optim.SGD(disc.parameters(), lr=args.learning_rate, momentum=0.5, nesterov=True)
        disc_public_opt = optim.SGD(list(pub.parameters())+list(act.parameters()),lr=args.learning_rate, momentum=0.5, nesterov=True)
        # disc_activation_opt = optim.SGD(act.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        g_opt = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=[0.5,0.999])
        disc_private_opt = crypten.optim.Adam(disc.parameters(), lr=args.learning_rate,betas=[0.5,0.999])
        # disc_activation_opt = optim.Adam(act.parameters(),lr=args.learning_rate,betas=[0.5,0.999])
        disc_public_opt = optim.Adam(list(pub.parameters())+list(act.parameters()), 
                                     lr=args.learning_rate,betas=[0.5,0.999])
    
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
    while iteration < args.iteration_number:
        for i, (img, _) in enumerate(dataloader):
            # record previous model to reload in case of wrong secure computation execution
            if not reload:
                try:
                    last_last_disc = last_disc.copy()
                    last_last_gen = last_gen.copy()
                    last_last_pub = last_pub.copy()
                except:
                    pass
                last_disc = disc.state_dict().copy()
                last_gen = gen.state_dict().copy()
                last_pub = pub.state_dict().copy()
            

            #================discriminator update=========
            # get fake output
            z = get_noise(args)
            fake_image = gen(z)
            fake_image_detach            = fake_image.detach()
            fake_image_enc               = crypten.cryptensor(fake_image_detach).cpu()
            fake_intermediate_output_enc = disc(fake_image_enc)
            fake_intermediate_output     = fake_intermediate_output_enc.get_plain_text().to(args.device)
            
            fake_intermediate_output.requires_grad = True
            fake_output                  = pub(fake_intermediate_output)
            fake_output                  = fake_output.view(fake_output.shape[0],-1)
            fake_output                  = act(fake_output)

            # get real output
            
            real_image_enc               = crypten.cryptensor(img)
            real_intermediate_output_enc = disc(real_image_enc)
            real_intermediate_output     = real_intermediate_output_enc.get_plain_text().to(args.device)

            #interpolated output
            g_opt.zero_grad()
            disc_private_opt.zero_grad()
            disc_public_opt.zero_grad()
            alpha = crypten.rand(args.batch_size,1,1,1)
            interpolated = fake_image_enc*(1-alpha) + real_image_enc*alpha
            interpolated.requires_grad=True
            interpolated_intermediate_enc = disc(interpolated)
            interpolated_intermediate = interpolated_intermediate_enc.get_plain_text().to(args.device)
            interpolated_intermediate = interpolated_intermediate.detach()
            interpolated_intermediate.requires_grad=True
            interpolated_output = pub(interpolated_intermediate)
            interpolated_output = interpolated_output.view(interpolated_output.shape[0],-1)
            interpolated_output = act(interpolated_output)
            grad = torch.autograd.grad(outputs=interpolated_output,
                                    inputs=interpolated_intermediate,
                                    grad_outputs=torch.ones(interpolated_output.size()).to(args.device),
                                    create_graph=True, retain_graph=True)[0]
            grad_enc = crypten.cryptensor(grad).cpu()
            interpolated_intermediate_enc.backward(grad_enc)
            grad = interpolated.grad.get_plain_text()
            grad = grad.view(args.batch_size,-1)
            grad_norm = torch.sqrt(torch.sum(grad**2,dim=1)+1e-12)
            grad_penalty = args.gp_weight*((grad_norm-1)**2).mean()


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
            real_output                  = pub(real_intermediate_output)
            real_output                  = real_output.view(real_output.shape[0],-1)
            real_output                  = act(real_output)

            # calculate discriminator loss
            def get_disc_loss(real_output, fake_output):
                return fake_output.mean()-real_output.mean()
                # real_target = torch.ones_like(real_output)
                # fake_target = torch.zeros_like(fake_output)
                # return criterion(fake_output, fake_target) + criterion(real_output, real_target)

            disc_loss = get_disc_loss(real_output, fake_output) + grad_penalty
            # if disc_loss < 0.25: continue
            if disc_loss.abs().max() > 1000:
                print("Fall back due to overflow, disc fake")
                crypten._setup_prng(iteration+fault)
                gen.load_state_dict(last_last_gen)
                disc.load_state_dict(last_last_disc,strict=False)
                pub.load_state_dict(last_last_pub)
                # last_gen = last_last_gen
                # last_disc = last_last_disc
                # last_pub = last_last_pub
                fault += 1
                reload = True
                continue
            
            
            g_opt.zero_grad()
            disc_private_opt.zero_grad()
            disc_public_opt.zero_grad()
            fake_grad = torch.autograd.grad(disc_loss, fake_intermediate_output,retain_graph=True)[0]
            real_grad = torch.autograd.grad(disc_loss, real_intermediate_output,retain_graph=True)[0]
            fake_grad_enc = crypten.cryptensor(fake_grad).cpu()
            real_grad_enc = crypten.cryptensor(real_grad).cpu()

            # update public discriminator part gradient
            disc_loss.backward()
            
            # update private discriminator part gradient
            fake_intermediate_output_enc.backward(fake_grad_enc)
            real_intermediate_output_enc.backward(real_grad_enc)

            # update weight
            
            disc_public_opt.step()
            disc_private_opt.step()

            if i % args.n_critics==0:
                #==============generator update================
                if args.model_type == "fc":
                    img = img.reshape(-1,np.prod(img.shape[1:]))
                # print(img.max())
                # assert False
                z = get_noise(args)
                fake_image = gen(z)
                fake_image_enc = crypten.cryptensor(fake_image).cpu()
                fake_intermediate_output_enc = disc(fake_image_enc)
                fake_intermediate_output = fake_intermediate_output_enc.get_plain_text().to(args.device)
                
                fake_intermediate_output_d = fake_intermediate_output.detach()
                fake_intermediate_output_d.requires_grad=True
                # fake_intermediate_output_d.zero_grad()
                fake_pub_output = pub(fake_intermediate_output_d)
                
                # fake_pub_output.requires_grad = True
                fake_pub_output = fake_pub_output.view(fake_pub_output.shape[0],-1)
                fake_output = act(fake_pub_output)
                # print(fake_output)
                def get_gen_loss(output):
                    return -output.mean()
                    # target = torch.ones_like(output).to(args.device)
                    # return criterion(output,target)
                
                gen_loss = get_gen_loss(fake_output)
                if gen_loss.abs().max() > 1000:
                    print("Fall back due to overflow, gen", gen_loss.abs().max())
                    crypten._setup_prng(iteration+fault)
                    gen.load_state_dict(last_last_gen)
                    disc.load_state_dict(last_last_disc,strict=False)
                    pub.load_state_dict(last_last_pub)
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
                private_grad = torch.autograd.grad(gen_loss, fake_intermediate_output_d)[0].cpu()
                private_grad_enc = crypten.cryptensor(private_grad)

                # backward pass such gradient
                fake_intermediate_output_enc.backward(private_grad_enc)
                # get gradient of the whole generator model
                public_gen_gradient = fake_image_enc.grad.get_plain_text().to(args.device)
                # backward pass the public gradient. Now the generator has all needed gradients
                fake_image.backward(public_gen_gradient)

                g_opt.step()
                # if i%args.n_critics == 0:
                accumulate_losses['disc'].append(disc_loss)
                accumulate_losses['gen'].append(gen_loss.item())


            # print(list(disc.parameters())[0].grad.get_plain_text())
            # assert False
            if i%args.n_critics==0:
                pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
            if iteration % args.iteration_save == (args.iteration_save-1):
                # pbar.set_description("[Iteration #{}] gen loss: {:.2f}, disc loss: {:.2f}".format(iteration,sum(accumulate_losses["gen"])/len(accumulate_losses['gen']),sum(accumulate_losses["disc"])/len(accumulate_losses['disc'])))
                if rank==1:
                    fake_image = gen(fixed_z)
                    # ds = 0.5
                    # dm = 0.5
                    # fake_image = torch.clamp(fake_image * ds + dm, 0, 1)
                    if args.model_type=="fc":
                        fake_image = fake_image.reshape(-1,1,28,28)
                    torchvision.utils.save_image(fake_image, "experiments/{}/image/tmp_generated_{}.jpg".format(args.save_path,iteration))
                    fa.write(f'{iteration+1},{sum(accumulate_losses["gen"])/len(accumulate_losses["gen"])},{sum(accumulate_losses["disc"])/len(accumulate_losses["disc"])} \n')
                accumulate_losses = {"gen":[], "disc":[]} 
                torch.save(gen.state_dict(),f"experiments/{args.save_path}/model/gen_{iteration+1}.pth")
                crypten.save(disc.state_dict(),f"experiments/{args.save_path}/model/disc.encrypted.{iteration+1}.{rank}")
                torch.save(pub.state_dict(), f"experiments/{args.save_path}/model/pub_{iteration+1}.pth")
                torch.save(act.state_dict(), f"experiments/{args.save_path}/model/act_{iteration+1}.pth")
            iteration += 1
            pbar.update(1)
            reload = False
            if iteration >= args.iteration_number:
                break
    pbar.close()
    
    


    

    
