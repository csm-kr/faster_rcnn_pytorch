from torchvision.models import vgg16_bn, vgg16
import torch
import torch.nn as nn

vgg16_torchvision = vgg16()
image = torch.randn([1, 3, 500, 375]).to('cuda')
vgg16 = vgg16_bn().features.to('cuda')
print(vgg16(image).size())


vgg16_ = nn.Sequential(*list(vgg16_bn().features.children())[:-1]).to('cuda')
print(vgg16_)
print(vgg16_(image).size())

vgg16__ = nn.Sequential(*list(vgg16_bn().features._modules.values())[:-1]).to('cuda')
print(vgg16__)
print(vgg16__(image).size())
print(vgg16_torchvision)