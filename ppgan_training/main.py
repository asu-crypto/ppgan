from random import seed
import crypten
from options import options
from model import *
from train_functions import *
from multiprocess_launcher import MultiProcessLauncher
from util import *

if __name__=="__main__":
    from crypten.config import cfg

    cfg.encoder.precision_bits = 21
    args = options().parse_args()
    seed_everything(args.seed)
    print(args.dataset)
    if not os.path.exists(f"experiments/{args.save_path}"):
        os.makedirs(f"experiments/{args.save_path}/image")
        os.makedirs(f"experiments/{args.save_path}/model")
    if args.model == "normal":
        import time
        t = time.time()
        train_normal(args)
        print("total running time:", time.time()-t)
    elif args.model=="sim":
        train_hybrid_sim(args)
    else:
        import time
        t = time.time()
        launcher = MultiProcessLauncher(2, train_secure, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
        print("total running time:", time.time()-t)