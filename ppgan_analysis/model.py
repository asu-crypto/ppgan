import torch
from torch import nn
import torchvision
from torchvision.models.resnet import Bottleneck, BasicBlock
import numpy as np
import torch.nn.functional as F

class ModelGenerator:
    def __init__(self):
        pass
    # @staticmethod
    def get_model(self,model_name, *args):
        # print(model_name, args)
        if model_name=="lr":
            return LR_Disc(*args)
        elif model_name=="mlp":
            return MLP_Disc(*args)
        elif model_name=="cnn":
            return CNN_Disc(*args)
        

class LR_Disc(nn.Module):
    # one layer perceptron discriminator
    def __init__(self, infan, outfan):
        super().__init__()
        self.linear = nn.Linear(infan, outfan)
        self.infan = infan
        self.outfan = outfan
        self.activation = nn.Sigmoid()
    
    def forward(self,x):
        return self.activation(self.linear(x))

class MLP_Disc(nn.Module):
    def __init__(self, num_sec_layers, *layers_shapes):
        super().__init__()
        self.num_sec_layers = num_sec_layers
        self.public_layers_shapes = layers_shapes[self.num_sec_layers:]
        if self.num_sec_layers >= 1:
            self.secured_layers_shapes = layers_shapes[:self.num_sec_layers+1]
            
        else:
            self.secured_layers_shapes = []#layers_shapes
        
        if len(self.secured_layers_shapes)==2:
            self.secured_layers = nn.Linear(self.secured_layers_shapes[0],self.secured_layers_shapes[1])
        elif len(self.secured_layers_shapes) < 2:
            self.secured_layers = lambda x: x
        else:
            self.secured_layers = nn.Sequential(
                nn.Linear(self.secured_layers_shapes[0], self.secured_layers_shapes[1]),
                *[nn.Sequential(nn.LeakyReLU(0.2),nn.Linear(self.secured_layers_shapes[i],self.secured_layers_shapes[i+1])) for i in range(1,len(self.secured_layers_shapes)-1)]
            )
        if len(self.public_layers_shapes)==2:
            self.public_layers = nn.Linear(self.public_layers_shapes[0],self.public_layers_shapes[1])
        else:
            self.public_layers = nn.Sequential(
                nn.Linear(self.public_layers_shapes[0], self.public_layers_shapes[1]),
                *[nn.Sequential(nn.LeakyReLU(0.2),nn.Linear(self.public_layers_shapes[i],self.public_layers_shapes[i+1])) for i in range(1,len(self.public_layers_shapes)-1)]
            )
        self.activation = nn.Sigmoid()
    def forward(self,x):
        x = x.reshape(x.shape[0],np.prod(x.shape[1:]))
        return self.activation(self.public_layers(self.secured_layers(x)))

class CNN_Disc(nn.Module):
    def __init__(self, num_sec_layers, *layers_shapes):
        # layers_shapes be list of elements of form (inchannel, outchannel, kernel size, stride)
        super().__init__()
        self.nc = num_sec_layers
        
        
        # if num_sec_layers >= 1:
        self.secured_layer_shapes = layers_shapes[:num_sec_layers]
        self.public_layer_shapes = layers_shapes[num_sec_layers:]
        # else:
        #     self.secured_layer_shapes = []
        #     self.public_la
        if num_sec_layers > 0:
            self.secured_layers = nn.Sequential(
                nn.Conv2d(in_channels=self.secured_layer_shapes[0][0],out_channels=self.secured_layer_shapes[0][1], 
                            kernel_size=self.secured_layer_shapes[0][2], stride=self.secured_layer_shapes[0][3],
                            padding=self.secured_layer_shapes[0][4]),
                *[nn.Sequential(nn.LeakyReLU(0.2),nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s,padding=p)) for (i,o,k,s,p) in self.secured_layer_shapes[1:]]
            )
        else:
            self.secured_layers = lambda x: x
        if len(self.public_layer_shapes) > 0:
            self.public_layers = nn.Sequential(
                nn.Conv2d(in_channels=self.public_layer_shapes[0][0],out_channels=self.public_layer_shapes[0][1], 
                kernel_size=self.public_layer_shapes[0][2], stride=self.public_layer_shapes[0][3],
                padding=self.public_layer_shapes[0][4]),
                *[nn.Sequential(nn.LeakyReLU(0.2),nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s,padding=p)) for (i,o,k,s,p) in self.public_layer_shapes[1:]]
            )
        else:
            self.public_layers = lambda x: x
        
        self.activation = nn.Sigmoid()
    def forward(self,x):
        y = self.activation(self.public_layers(self.secured_layers(x)))
        return y.reshape(-1)  
     
class StyleGan_Disc(nn.Module):
    def __init__(self,in_channels,input_size):
        pass

class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='LeakyReLU')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def weight_count(model):
    count = 0
    for param in model.parameters():
        count += np.prod(param.size())
    return count

# sim_distance
def sim_dist(model_weight_a,model_weight_b):
    total_dist = 0
    for weight_a, weight_b in zip(model_weight_a, model_weight_b):
        cost = (weight_a*weight_b).sum()
        cost = 1 - cost / (weight_a.square().sum()+1e-20).sqrt() / (weight_b.square().sum()+1e-20).sqrt()
        total_dist += cost.item()
    return total_dist/len(model_weight_a)
    
# mse distance
def mse_dist(model_weight_a, model_weight_b):
    total_dist = 0
    for weight_a, weight_b in zip(model_weight_a, model_weight_b):
        cost = (weight_a-weight_b).square().sum()
        cost = (cost+1e-20).sqrt()
        total_dist += cost.item()
    return total_dist/len(model_weight_a)

# mae distance
def mae_dist(model_weight_a, model_weight_b):
    total_dist = 0
    for weight_a, weight_b in zip(model_weight_a, model_weight_b):
        cost = (weight_a-weight_b).abs().mean()
        total_dist += cost.item()
    return total_dist/len(model_weight_a)

class DCGenerator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(DCGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.leaky_LeakyReLU(self.deconv1(input))
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)), 0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)), 0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)), 0.2)
        x = F.tanh(self.deconv5(x))

        return x



if __name__=="__main__":
    pass