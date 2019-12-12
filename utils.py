from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import copy
from model import *
from torch.autograd import Variable
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' image loading part '''
# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 256  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


''' image plotting '''
unloader = transforms.ToPILImage()  # reconvert into PIL image
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def get_input_optimizer(input_img, optimizer = "SGD"):
    # this line to show that input is a parameter that requires a gradient
    if optimizer == "LBFGS":
        print("we use {0} to optimize...".format(optimizer))
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    else:
        print("we use {0} to optimize...".format(optimizer))
        optimizer = optim.SGD([input_img.requires_grad_()], lr=0.01, momentum=0.9)
    return optimizer


''' load order dictionary '''
def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        return patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


''' gan generator evaluation '''
def gan_generator_eval(model, input, output):
    input_img = Image.open(input)
    input_img = input_img.resize((imsize, imsize))
    if model == "shuimo":
        gan_weights = torch.load("./Shuimo-Generator.pth")
    else: #Gongbi
        gan_weights = torch.load("./Gongbi-Generator.pth")
    gan = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
    for key in list(gan_weights.keys()):
        patch_instance_norm_state_dict(gan_weights, gan, key.split('.'))
    gan.load_state_dict(gan_weights)
    gan.eval()

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    tr = transforms.Compose(transforms_)
    input_A = torch.Tensor(1, 3, 256, 256)
    real_A = Variable(input_A.copy_(tr(input_img)))
    fake_B = 0.5 * (gan(real_A).data + 1.0)
    save_image(fake_B, output)
    output_img = image_loader(output)
    return (output_img, gan.model)


