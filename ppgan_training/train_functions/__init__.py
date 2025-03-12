from model import *
from dataset import *
import crypten
from train_functions.train_hybrid_wgangp import train_hybrid_wgan
from .train_normal import train_normal
from .train_half import train_half
from .train_full import train_full
from .train_hybrid import train_hybrid
from .train_hybrid_sim import train_hybrid_sim
from .train_hybrid_wgangp import train_hybrid_wgan
from .train_hybrid_cgan import train_hybrid_cgan
from .train_half_cgan import train_half_cgan
def train_secure(args):
    # if args.num_secure < 1:
    #     raise ValueError
    if args.model_type == "fc":
        if args.dataset.lower() == "mnist":
            gen = FC_Generator(args)#MLP_Disc(True,args.activation_type, 0, args.zdim, 256, 512, 1024, 784)#MLP_Disc(True, args.activation_type, 0, args.zdim, 256, 512, 1024, 784)#FC_Generator(args)#
            # disc= MLP_Disc(False,args.activation_type, args.num_secure, 784, 1024, 512, 256, 1)
            disc= MLP_Disc(False,args.activation_type, args.num_secure, 784, 512, 256, 128, 1)
            sample_gen = torch.rand(args.batch_size, args.zdim)
            sample_disc = torch.rand(args.batch_size, 784)
        else:
            raise NotImplementedErrord
    elif args.model_type == "cnn":
        if args.dataset.lower() == "mnist":
            gen = DCGenerator(args)
            disc = CNN_Disc(args.activation_type, args.num_secure, (1,16,3,2,1),(16,32,3,2,1),(32,64,3,2,1),(64,128,3,2,1))
            sample_gen = torch.rand(args.batch_size, args.zdim)
            sample_disc = torch.rand(args.batch_size, 1, 64, 64)
        elif args.dataset.lower() == "celeba" or args.dataset.lower()=="cifar10":
            gen = DCGenerator(args)
            disc = CNN_Disc(args.activation_type, args.num_secure, (3,16,3,2,1),(16,32,3,2,1),(32,64,3,2,1),(64,128,3,2,1))
            sample_gen = torch.rand(args.batch_size, args.zdim)
            sample_disc = torch.rand(args.batch_size, 3, 64, 64)
    else:
        raise NotImplementedError
    
    if args.model == "full":
        torch.onnx.export(
            disc,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        # torch.onnx.export(
        #     gen,
        #     (sample_gen,),
        #     'tmp_gen.onnx'
        # )
        if args.model_type=="fc":
            gen_enc = FCGenerator(args)
            disc_enc = FCDiscriminator()
        else:
            gen_enc = SecureGenerator(args)
            disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        # gen_enc = crypten.nn.from_pytorch(gen, sample_gen)
        # gen_enc = crypten.nn.from_onnx(open("tmp_gen.onnx","rb"))
        disc_enc.encrypt()
        gen_enc.encrypt()
        train_full(gen_enc, disc_enc, args)
        # disc_enc = crypten.nn.from_pytorch(disc, sample_disc)
    elif args.model == "half":
        gen_enc = gen
        torch.onnx.export(
            disc,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        disc_enc.encrypt()
        train_half(gen_enc, disc_enc, args)
    elif args.model=="half_cgan":
        args.zdim += 10
        gen_enc = FC_Generator(args)
        args.zdim -= 10
        disc = MLP_Disc(False,args.activation_type, args.num_secure, 794, 512, 256, 128, 1)
        sample_disc =  torch.rand(args.batch_size, 794)
        torch.onnx.export(
            disc,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        disc_enc.encrypt()
        train_half_cgan(gen_enc, disc_enc, args)
    elif args.model == "hybrid":
        gen_enc = gen
        disc_private = disc.secured_layers 
        disc_public = disc.public_layers
        # disc_activation = disc.activation
        # disc_enc = disc_private 
        torch.onnx.export(
            disc_private,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        # torch.onnx.export(
        #     disc,
        #     (sample_disc,),
        #     'tmp_full.onnx'
        # )
        disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        # disc_full = crypten.nn.from_onnx(open("tmp_full.onnx","rb"))
        # disc_enc = disc_full.secured_layers
        disc_enc.encrypt()
        train_hybrid(gen_enc, disc_enc, disc_public, args)
    
    elif args.model == "cgan":
        args.zdim += 10
        gen = FC_Generator(args)
        args.zdim -= 10
        disc = MLP_Disc(False,args.activation_type, args.num_secure, 794, 512, 256, 128, 1)
        sample_disc =  torch.rand(args.batch_size, 794)
        gen_enc = gen
        disc_private = disc.secured_layers 
        disc_public = disc.public_layers
        # disc_activation = disc.activation
        # disc_enc = disc_private 
        torch.onnx.export(
            disc_private,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        # torch.onnx.export(
        #     disc,
        #     (sample_disc,),
        #     'tmp_full.onnx'
        # )
        disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        # disc_full = crypten.nn.from_onnx(open("tmp_full.onnx","rb"))
        # disc_enc = disc_full.secured_layers
        disc_enc.encrypt()
        train_hybrid_cgan(gen_enc, disc_enc, disc_public, args)
    elif args.model == "hybridWGAN":
        gen_enc = gen
        disc_private = disc.secured_layers 
        disc_public = disc.public_layers
        # disc_activation = disc.activation
        # disc_enc = disc_private 
        torch.onnx.export(
            disc_private,
            (sample_disc,),
            'tmp_disc.onnx'
        )
        disc_enc = crypten.nn.from_onnx(open("tmp_disc.onnx","rb"))
        disc_enc.encrypt()
        train_hybrid_wgan(gen_enc, disc_enc, disc_public, args)
    # elif args.model == "sim":
    #     gen_enc = gen
    #     disc_private = disc.secured_layers 
    #     disc_public = disc.public_layers
    #     disc_activation = disc.activation
    #     train_hybrid_sim(gen_enc, disc_private, disc_public, disc_activation, args)
    # test Sep 14
    # unit_test = crypten.rand(4,*sample_disc[1:].shape)
    # disc_enc.encrypt()
    # y = disc_enc(unit_test)
    # print(y.shape)
    # if args.model == "hybrid":
    #     z = disc_public(y.get_plain_text())
    #     out = disc_activation(z)
    #     print(out)
    
    # train base on args model




__all__ = ['train_normal','train_secure','train_hybrid_sim']