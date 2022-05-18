['''LeNet in PyTorch.''']
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from visdom import Visdom
import time
from models.struct_conv_orient import *


class Struct_Conv(nn.Module):
    def __init__(self,in_channel,out_channel,rin=4,rout=4,size=5,r=2.,kernel_size=(1,1),padding=(2,2),stride=1,device='cuda',logger=None,ifbias=False,ifbn=True):
        super(Struct_Conv,self).__init__()
        self.rin = rin
        self.rout = rout
        self.ifbn=ifbn
        self.conv = nn.Conv2d(int(in_channel / self.rin), int(out_channel / self.rout), kernel_size,bias=False)
        # if self.ifbn:
        self.bn = nn.BatchNorm2d(int(out_channel / self.rout),momentum=0.1/self.rout)
        self.padding = padding
        self.stride = stride
        self.ifbias = ifbias
        oc = int(out_channel / self.rout) 
        ic = int(in_channel / self.rin)
        ks = kernel_size
        if ks[1] == 1:
            p_valid = False
        else:
            p_valid = True 
        print('p_valid:',ks[1],p_valid)
        bestr = 1.
        self.theta1 = torch.Tensor([2*np.pi/self.rin]).to(device)
        self.theta2 = torch.Tensor([2*np.pi/self.rout]).to(device)
        self.wr = nn.Parameter((torch.Tensor(
            1,oc,1, ic,ks[0],ks[1],1,1
            ).uniform_(-np.pi,np.pi)).to(device),requires_grad=True)
        if p_valid:
            theta = self.theta2
            sp = 1.
            X,Y = torch.meshgrid(torch.linspace(-(ks[0]-1)/2*sp,(ks[0]-1)/2*sp,ks[0]),
                                torch.linspace(-(ks[1]-1)/2*sp,(ks[1]-1)/2*sp,ks[1]))
            self.wpx,self.wpy = X.reshape(1,1,1, 1,ks[0],ks[1],1,1).to(device),Y.reshape(1,1,1, 1,ks[0],ks[1],1,1).to(device)
            d = ((self.wpx.max()-self.wpx.min()).pow(2)+(self.wpy.max()-self.wpy.min()).pow(2)).sqrt()
            # self.var_position = torch.max(torch.Tensor([mind]).to(device),bestr*d*np.sqrt(2)*((theta/2).sin()))
            self.var_position = nn.Parameter(torch.Tensor(
                np.array([1.*np.sqrt(2)])),requires_grad=False).to(device)
            self.var_rotate = nn.Parameter(torch.Tensor(
                np.array([2.*np.pi/self.rin*bestr*np.sqrt(2.)])),requires_grad=False).to(device)
        else:
            self.wpx = nn.Parameter((torch.Tensor(
                1,oc,1, ic,ks[0],ks[1],1,1
                ).uniform_(-(r-0.5),(r-0.5))).to(device),requires_grad=True)
            self.wpy = nn.Parameter((torch.Tensor(
                1,oc,1, ic,ks[0],ks[1],1,1
                ).uniform_(-(r-0.5),(r-0.5))).to(device),requires_grad=True)
            self.var_position = nn.Parameter((torch.Tensor(
                1,oc,1, ic,ks[0],ks[1],1,1
                ).uniform_(bestr*np.sqrt(2.),bestr*np.sqrt(2.))).to(device),requires_grad=False)
            if self.rin==1:
                self.var_rotate = nn.Parameter(torch.Tensor(
                    np.array([2.*np.pi/self.rout*bestr*np.sqrt(2.)])),requires_grad=False).to(device)
                print('rin var_rotate')
            else:
                self.var_rotate = nn.Parameter(torch.Tensor(
                    np.array([2.*np.pi/self.rin*bestr*np.sqrt(2.)])),requires_grad=False).to(device)
                print('rout var_rotate')
        self.size,self.r = size,r
        self.out_c = out_channel
        self.in_c = in_channel
        self.size = size
        kx = np.array([[i for i in range(size)]]*size)-r
        ky = kx.transpose()
        self.kx = torch.Tensor(kx).reshape(1,1,1,1,1,1,size,size).to(device)
        self.ky = torch.Tensor(ky).reshape(1,1,1,1,1,1,size,size).to(device)
        self.tpi = torch.Tensor([np.pi*2]).to(device)
        self.e = torch.Tensor(np.array([1e-5])).to(device)
        self.iin = torch.Tensor([i for i in range(self.rin)]).reshape(1,1,-1,1,1,1,1,1).to(device)*self.theta1
        self.iout = torch.Tensor([i for i in range(self.rout)]).reshape(-1,1,1,1,1,1,1,1).to(device)*self.theta2

    def forward(self,x):
        out = F.conv2d(x,self.get_weight(),padding=self.padding,stride=self.stride)
        if self.ifbn:
            out = G.BN(out,self.bn,r=self.rout)
        else:
            b = self.bn.bias.reshape(1,-1,1,1)
            b = [b for _ in range(self.rout)]
            out = out + torch.cat(b,dim=1)
        # print(out.size())
        return out
    
    def R(self,x,y,theta,ifnp=False):
        if not ifnp:
            c = theta.cos()
            s = theta.sin()
        else:
            c = np.cos(theta)
            s = np.sin(theta)
        return c*x-s*y,s*x+c*y
    
    def get_weight(self,ifsep=False):
        
        w1 = self.conv.weight.unsqueeze(1).unsqueeze(0).unsqueeze(6).unsqueeze(6)
        # print('w1:',w1.size())
        wpx,wpy = self.wpx,self.wpy
        wpx = wpx-wpx.mean(dim=(4,5),keepdim=True)
        wpy = wpy-wpy.mean(dim=(4,5),keepdim=True)
        var_p = self.var_position

        wpx,wpy = self.R(wpx,wpy,self.iout)
        a,b = self.kx-wpx,self.ky-wpy
        w = (-((a/var_p).pow(2)+(b/var_p).pow(2))).exp()
        # print('w:',w.size())
        # w = w/(w.sum(dim=(6,7),keepdim=True)+self.e) * w1
        if self.rin != 1:
            wr1 = self.wr-self.iin +self.iout
            l1 = torch.min(wr1%self.tpi,(-wr1)%self.tpi)
            l1 = (-(l1/self.var_rotate).pow(2)).exp()
            w = w*l1

        # print('w:',w.size())
        w = w/(w.sum(dim=(2,6,7),keepdim=True)+self.e) * w1
        if not ifsep:
            w = w.sum(dim=(4,5)).reshape(self.out_c,self.in_c,self.size,self.size)
            # print('final w:',w.size())

        return w


class MyLeNetStructInvariantNew_nms(nn.Module):
    def __init__(self,logger=None):
        super(MyLeNetStructInvariantNew_nms, self).__init__()
        self.r = 16
        r = self.r
        base = 160
        ks = (3,1)
        self.rconv1 = Struct_Conv(1,base,1,r,kernel_size=ks,logger=logger)
        self.rconv2 = Struct_Conv(base,base,r,r,kernel_size=ks,logger=logger)
        self.rconv3 = Struct_Conv(base,base,r,r,kernel_size=ks,logger=logger)
        self.rconv4 = Struct_Conv(base,base,r,r,kernel_size=ks,logger=logger)
        self.rconv5 = Struct_Conv(base,base,r,r,kernel_size=ks,logger=logger)
        self.rconv6 = Struct_Conv(base,base,r,r,kernel_size=ks,logger=logger)

        self.rconv7 = Struct_Conv(base,10*r,r,r,ifbn=False,kernel_size=ks,logger=logger)

        # self.r = 4
        # r = self.r
        # base = 40
        # self.rconv1 = Rotate_Conv(3,base,kernel_size=(3,3),padding=(1,1),sep=True)
        # self.rconv2 = Rotate_Conv(base,base)
        # self.rconv3 = Rotate_Conv(base,base)
        # self.rconv4 = Rotate_Conv(base,base)
        # self.rconv5 = Rotate_Conv(base,base)
        # self.rconv6 = Rotate_Conv(base,base)
        # self.rconv7 = Rotate_Conv(base, 40, ifbn=False)


        # base = 20
        # self.rconv1 = Norm_Conv(1,base)
        # self.rconv2 = Norm_Conv(base,base)
        # self.rconv3 = Norm_Conv(base,base)
        # self.rconv4 = Norm_Conv(base,base)
        # self.rconv5 = Norm_Conv(base,base)
        # self.rconv6 = Norm_Conv(base,base)
        # self.rconv7 = Norm_Conv(base, 10, ifbn=False)


        # self.viz = Visdom(server='http://127.0.0.1', port=8097)
        # assert self.viz.check_connection()

    def forward(self,x):
        out = F.relu(self.rconv1(x))
        out = F.relu(self.rconv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.rconv3(out))
        out = F.relu(self.rconv4(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.rconv5(out))
        out = F.relu(self.rconv6(out))
        # a = np.random.randint(0,2,(2))
        # out = F.pad(out,(a[0],1-a[0],a[1],1-a[1]))
        # out = F.max_pool2d(out, 2)
        out = self.rconv7(out)
        out = self.rotatedim_max(out,self.r)

        out = F.max_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)

        return out,0*out.sum()

    def observe(self,x):
        print('rconv1:weight,wr,wpx,wpy',
            self.rconv1.conv.weight[0,:,:,:],
            self.rconv1.wr[0,:,:,:],
            self.rconv1.wpx[0,:,:,:],
            self.rconv1.wpy[0,:,:,:],
            self.rconv1.var_r[0,:,:,:],
            self.rconv1.var_x[0,:,:,:],
            self.rconv1.var_y[0,:,:,:])
        print(self.rconv1.conv.weight.size(),self.rconv1.wr.size())
        self.show_multi(self.rconv1.get_weight(ifsep=True)[2::5,0,:,:],ifneg=True)
        self.show_multi(self.rconv1.get_weight(ifsep=True)[2::5,1,:,:],ifneg=True)
        self.show_multi(self.rconv1.get_weight()[2::5,0,:,:],ifneg=True)
        self.show_multi(self.rconv1.get_weight()[0:5,0,:,:],ifneg=True)

        out = self.rconv1(x)
        out = F.relu(out)
        # out = F.max_pool2d(out, 2)
        # out = self.rconv3(out)
        # out = F.relu(out)
        self.show_multi(out[0,2::5,:,:],ifsum=True)
        self.show_multi(out[0,0:5,:,:])

    def subsample(self,x,p,s):
        return x[:,:,::s,::s],[p[0][:,:,::s,::s]/s,p[1][:,:,::s,::s]/s]

    def nonpolar_add(self,rot,non, r=4):
        a = rot.size(1)
        # print(x.size())
        rt = torch.split(rot, int(a / r), dim=1)
        rt = [rt[i]+non for i in range(r)]
        return torch.cat(rt,dim=1)

    def nms(self, x, r=4):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, r, int(s1 / r), s2, s3)
        y = torch.max(x, dim=1, keepdim=True)[0]
        y = torch.cat([y for i in range(r)], dim=1)
        y = y.ne(x).float()
        x = x * y
        return x.mean()

    def rotatedim_sum(self,x, r=4):
        s0, s1, s2, s3 = x.size()
        return x.view(s0, r, int(s1 / r), s2, s3).sum(dim=1)

    def rotatedim_max(self,x, r=4):
        s0,s1,s2,s3 = x.size()
        return x.view(s0,r,int(s1/r),s2,s3).max(dim=1)[0]

    def Max_pool(self,x,value,stride=2,padding=None):
        def orient_conv_max(weight,value,dim):
            index = weight.argmax(dim,keepdim=True)
            for i,a in enumerate(value):
                value[i] = value[i].gather(dim,index).squeeze(dim)
            weight = weight.gather(dim,index).squeeze(dim)
            return weight,value

        if padding is not None:
            x=F.pad(x,padding)
            value = [F.pad(value[i],padding) for i,a in enumerate(value)]
        s0,s1,s2,s3 = x.size()
        # print('x',x.size(),'stride',stride)
        a = x.view(s0,s1,int(s2/stride),stride,int(s3/stride),stride)
        v = [value[i].view(s0,s1,int(s2/stride),stride,int(s3/stride),stride) for i,a in enumerate(value)]
        a,v = orient_conv_max(a,v,3)
        a,v = orient_conv_max(a,v,4)
        return a,v

    def show_input(self, x):
        print('show_input')
        print(x.size())
        
        b = x.cpu().detach().numpy()
        bmin = np.min(b)
        bmax = np.max(b)
        self.viz.image((b-bmin)/(bmax-bmin))

    def show_multi(self,x,ifneg=False,ifsum=False):
        print('show_multi')
        print(x.size())

        b = x.cpu().detach().numpy()
        if ifneg:
            b = b-np.min(b)
        s1 = x.size(0)
        for i in range(s1):
            self.viz.image(b[i]/np.max(b))
        if ifsum:
            b = np.sum(b,axis=0)
            self.viz.image(b/np.max(b))