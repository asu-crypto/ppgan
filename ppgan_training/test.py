from random import seed
import crypten
import torch
from torch import nn
from util import seed_everything
from multiprocess_launcher import MultiProcessLauncher
import time
seed_everything()
# crypten.init()
# model = nn.Sequential(
#     nn.Linear(2,2),
#     nn.LeakyReLU()
# )
# x = torch.tensor([1.0,-1.0])
# y = model(x).sum()
# print(y)
# y.backward()
# for p in list(model.parameters()):
#     print(p.grad)
# torch.onnx.export(model,x,'tmp.onnx')
# model_enc = crypten.nn.from_onnx(open('tmp.onnx','rb')).encrypt()
# x_enc = crypten.cryptensor(x)
# yp = model_enc(x_enc).sum()
# yp.backward()
# for p in list(model_enc.parameters()):
#     print(p.grad.get_plain_text())
# # print(yp)

def main():
    x = crypten.rand(1)
    t=time.time()
    mask = x.gt(0.5)
    # m = mask.clone()
    print(mask.get_plain_text(),x.get_plain_text())
    print("Time",time.time()-t)
    
    # print(m.ptype)
    # print(((1000-m*1000)*0.01).get_plain_text())

if __name__=="__main__":
    
    
    launcher = MultiProcessLauncher(15, main)
    launcher.start()
    launcher.join()
    launcher.terminate()
    