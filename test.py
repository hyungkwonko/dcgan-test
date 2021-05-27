import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from DCGAN_w_linear import Generator

ngpu = 4
nz = 100
device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

if __name__ == '__main__':
    netG = Generator(ngpu)

    SAVE_PATH = './model/netG.pkl'
    torch.load(netG, SAVE_PATH)

    print(netG)


