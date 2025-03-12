import argparse

def options():
    parser = argparse.ArgumentParser("ppgan attack")
    parser.add_argument("-ns","--num-secure", type=int, default=1, help="Number of secure layer")
    parser.add_argument("-as","--attack-side", type=str, choices=["disc","gen"], default="disc",
                        help="Choose between discriminator or generator to implement model recovery attack")
    parser.add_argument("-bsf","--batch-size-fake", type=int, default=16, help="Batch size allow to be queried")
    parser.add_argument("-ds","--dataset", type=str, choices=["mnist","cifar10","celeba"],help="Dataset the model trained on")
    parser.add_argument("--trained_model", action="store_true", help="whether the model is pretrained")
    parser.add_argument("-bsr","--batch-size-real", type=int, default=4, help="Batch size of real image data")
    parser.add_argument("-nri","--num-recon-iteration",type=int, default=20000, help="Number of iterations for weight recovery")
    parser.add_argument("--seed",type=int,default=1024,help="Random seed for the system reproducibility")
    return parser

if __name__=="__main__":
    args = options().parse_args()
    print(args.num_secure)