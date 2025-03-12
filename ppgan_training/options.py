import argparse

def options():
    parser = argparse.ArgumentParser("PPGAN Configuration")
    parser.add_argument("-ds", "--dataset", default="mnist", type=str,help="Dataset we train the model on")
    parser.add_argument("-m", "--model", choices=["full","hybrid","half","normal","sim","cgan","half_cgan","hybridWGAN"])
    parser.add_argument("-bs", "--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("-ns", "--num-secure", type=int, default=0, help="Number of secure layer in discriminator, -1 for fully secure")
    parser.add_argument("-mt", "--model-type", choices=["fc","cnn"],help="type of model to be trained on, fully connected or deep convolution")
    parser.add_argument("-at", "--activation-type", default="relu",choices=["relu","leaky"], help="activation function, we choose between relu and leaky relu")
    parser.add_argument("-it","--iteration-number", type=int, default=100000, help="Number of iteration for training")
    parser.add_argument("-its","--iteration-save", type=int,default=1000, help="Number of iteration per saving")
    parser.add_argument("-z", "--zdim", type=int, default=100, help="Dimension of noise")
    parser.add_argument("-lr","--learning-rate", default=0.01,type=float,help="Learning rate for GAN")
    parser.add_argument("--momentum_G",type=float,default=0.4,help="momentum for SGD optimizer")
    parser.add_argument("--momentum_D",type=float,default=0.4,help="momentum for SGD optimizer")
    parser.add_argument("-opt","--optimizer", default="sgd",type=str,choices=["adam","sgd"],help="type of optimizer")
    parser.add_argument("--device", default="cuda",choices=["cuda","cpu"])
    parser.add_argument("-p", "--save-path", type=str, default="./", help="Path to save the data")
    parser.add_argument('--seed', type=int, default=1024 ,help='random seed for experiment')
    parser.add_argument("-c",'--continue-from', type=int, default=-1, help="iteration to continue training from")
    parser.add_argument("--gp-weight",type=float,default=10,help="Gradient penalty weight")
    parser.add_argument("--n-critics",type=int,default=5,help="Number of critic training iteration per generator training iteration")
    return parser