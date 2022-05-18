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


class G(object):
    @staticmethod
    def rot(v1,v2,theta,sep=False):
        c = np.cos(theta)
        s = np.sin(theta)
        # print('rot',c.size(),'ifseperate',ifseperate,len(theta))

        if sep:
            s0,s1,s2,s3 = v1.size()
            c = torch.Tensor(c).view(1,s1,1,1).cuda()
            s = torch.Tensor(s).view(1,s1,1,1).cuda()
            V1 = c*v1-s*v2
            V2 = s*v1+c*v2
        elif len(theta)==1:
            V1 = c*v1-s*v2
            V2 = s*v1+c*v2

        else:
            V1 = []
            V2 = []

            for i in range(len(c)):
                V1.append(c[i]*v1-s[i]*v2)
                V2.append(s[i]*v1+c[i]*v2)
            V1 = np.array(V1)
            V2 = np.array(V2)

        return V1,V2

    @staticmethod
    def normal_2d_kernel(self,size,theta,var_x,var_y):
        # print('normal_2d_kernel')

        r = int((size-1)/2)
        # d_theta = np.array(info['d_theta'])
        theta = np.array(theta)
        var_x = np.array(var_x).reshape(-1,1,1)
        var_y = np.array(var_y).reshape(-1,1,1)
        
        kx = np.array([[i for i in range(size)]]*size)-r
        ky = kx.transpose()
        # print('kx,ky',kx,ky)
        Kx,Ky = G.rot(kx,ky,theta)
        Kx = Kx/var_x
        Ky = Ky/var_y
        # print('Kx,Ky',Kx,Ky)
        kernel = np.exp(-(Kx**2+Ky**2)/2)
        # print('kernel',kernel)
        k = np.sum(kernel,axis=(1,2),keepdims=True)
        kernel /= k

        kernel = torch.Tensor(kernel).cuda()
        kernel = kernel.unsqueeze(1)

        return kernel
    
    @staticmethod
    def Laplas_kernel(size,theta):
        info1={'var_x':np.ones(len(theta))*1.,'var_y':np.ones(len(theta))*3.}
        info2={'var_x':np.ones(len(theta))*1.5,'var_y':np.ones(len(theta))*3.}
        k1 = G.normal_2d_kernel(size,theta,info1['var_x'],info1['var_y'])
        k2 = G.normal_2d_kernel(size,theta,info2['var_x'],info2['var_y'])
        return k1-k2

    @staticmethod
    def Lateral_kernel(size,theta):
        info1={'var_x':np.ones(len(theta))*1.,'var_y':np.ones(len(theta))*3.}
        info2={'var_x':np.ones(len(theta))*1.5,'var_y':np.ones(len(theta))*3.}
        k1 = G.normal_2d_kernel(size,theta,info1['var_x'],info1['var_y'])
        k1/=k1.max()
        k2 = G.normal_2d_kernel(size,theta,info2['var_x'],info2['var_y'])
        k2/=k2.max()
        k = k1-k2
        return k/k.mean()

    @staticmethod
    def Laplas_conv(x,kernel,pad=(2,2)):
        s = x.size()
        out = F.conv2d(x.reshape(-1,1,s[2],s[3]),kernel,padding=pad)
        return out.reshape(s)

    @staticmethod
    def Span_conv(x,kernel,pad=[2,2]):
        s = x.size()
        out = F.conv_transpose2d(x.reshape(-1,1,s[2],s[3]),kernel)[:,:,2:-2,2:-2]
        return out.reshape(s)

    @staticmethod
    def Laplas_loss(x,kernel,pad=(2,2)):
        s = x.size()
        out = F.conv2d(x.reshape(-1,1,s[2],s[3]),kernel,padding=pad)*(x.reshape(-1,1,s[2],s[3]))
        return out.reshape(s).sum(1).mean()

    @staticmethod
    def normal_2d_kernel(size,theta,var_x,var_y):
        # print('normal_2d_kernel')

        r = int((size-1)/2)
        # d_theta = np.array(info['d_theta'])
        theta = np.array(theta)
        var_x = np.array(var_x).reshape(-1,1,1)
        var_y = np.array(var_y).reshape(-1,1,1)

        
        kx = np.array([[i for i in range(size)]]*size)-r
        ky = kx.transpose()
        # print('kx,ky',kx,ky)
        Kx,Ky = G.rot(kx,ky,theta)
        Kx = Kx/var_x
        Ky = Ky/var_y
        # print('Kx,Ky',Kx,Ky)
        kernel = np.exp(-(Kx**2+Ky**2)/2)
        # print('kernel',kernel)
        k = np.sum(kernel,axis=(1,2),keepdims=True)
        kernel /= k

        kernel = torch.Tensor(kernel).cuda()
        kernel = kernel.unsqueeze(1)
        print(kernel)

        return kernel

    @staticmethod
    def orient_max(x,ifx='x'):
        xp = F.pad(x,(1,1,1,1))
        if 'x' in ifx:
            x = x*torch.gt(xp[:,:,1:-1,1:-1],xp[:,:,1:-1,0:-2])*torch.gt(xp[:,:,1:-1,1:-1],xp[:,:,1:-1,2:])
        if 'y' in ifx:
            x = x*torch.gt(xp[:,:,1:-1,1:-1],xp[:,:,0:-2,1:-1])*torch.gt(xp[:,:,1:-1,1:-1],xp[:,:,2:,1:-1])
        return x

    @staticmethod
    def BN(x, bn,r=4):
        # print('BN')
        a = x.size(1)
        # print(a)
        x = torch.split(x, int(a / r), dim=1)
        x = [bn(x[i]) for i in range(r)]
        x = torch.cat(x, dim=1)
        # print(x.size())
        return x


    @staticmethod
    def deconv_kernel(size,theta,ifcorner=False):
        r = int((size-1)/2)
        # d_theta = np.array(info['d_theta'])
        theta = np.array(theta)

        kx = -(np.array([[i for i in range(size)]]*size)-r)
        ky = kx.transpose()

        # Kd= G.distance(kx,ky,theta)

        dist_x = torch.Tensor(kx).reshape(1,1,5,5).cuda()
        dist_y = torch.Tensor(ky).reshape(1,1,5,5).cuda()
        # kernel_dist = kernel.unsqueeze(1)
        if ifcorner:
            kernel_weight = G.normal_2d_kernel(size,theta,2.3,2.3)
        else:
            kernel_weight = G.normal_2d_kernel(size,theta,2.3,1.0)

        return kernel_weight,[dist_x,dist_y]

    @staticmethod
    def attractor(x,kernel_weight,kernel_dist):
        s = x.size()
        x = x.reshape(-1,1,s[2],s[3])
        x_bool = torch.gt(x,0).float()
        out_weight = F.conv_transpose2d(x,kernel_weight)[:,:,2:-2,2:-2]
        out_dist_x = F.conv_transpose2d(x,kernel_dist[0]*kernel_weight)[:,:,2:-2,2:-2]
        out_dist_y = F.conv_transpose2d(x,kernel_dist[1]*kernel_weight)[:,:,2:-2,2:-2]
        out = F.conv_transpose2d(x_bool,kernel_weight)[:,:,2:-2,2:-2]
        # out = out/(out_weight+1e-5)
        out = out_weight/(out+1e-5)
        out_dist_x /= (out_weight+1e-5)
        out_dist_y /= (out_weight+1e-5)
        # print('kernel weight dist',kernel_weight.size(),kernel_dist.size())
        # print('attractor',s,x.size(),'out,weight,dist',out.size(),out_weight.size(),out_dist.size())

        return out.reshape(s),out_weight.reshape(s),out_dist_x.reshape(s),out_dist_y.reshape(s)

    @staticmethod
    def arg_attractor(f,p):
        px = p[0]
        py = p[1]
        px = px.round()
        py = py.round()
        f_out = torch.zeros(f.size()).cuda()
        f_out = F.pad(f_out,(3,4,3,4))
        print('f_out size',f_out.size(),'f size', f.size())
        for i in range(7):
            for j in range(7):
                f_out[:,:,j:-(7-j),i:-(7-i)] += f *py.eq(j-3)* px.eq(i-3)

        return f_out

# k = G.Lateral_kernel(5,np.array([0.]))
# dk = G.normal_2d_kernel(5,np.array([0.]),2.5,1)


class Lateral(object):
    def __init__(self):
        # super(Lateral,self)
        self.r = 4
        self.kw,self.kd = G.deconv_kernel(5,[0.])
        self.kw =Variable(self.kw,requires_grad=False)
        self.kd =[Variable(self.kd[0],requires_grad=False),Variable(self.kd[1],requires_grad=False)]
        self.kwc,self.kdc = G.deconv_kernel(5,[0.],ifcorner=True)
        self.kwc =Variable(self.kwc,requires_grad=False)
        self.kdc =[Variable(self.kdc[0],requires_grad=False),Variable(self.kdc[1],requires_grad=False)]

    def forward(self,x,ifcorner=False):
        out = []
        out_dist_x = []
        out_dist_y = []
        out_weight = []
        out1 = torch.split(x,int(x.size(1)/self.r),dim=1)
        if ifcorner:
            kw = self.kwc
            kd = self.kdc
        else:
            kw = self.kw
            kd = self.kd
        for i,a in enumerate(out1):
            if i!=0:
                kw = kw.permute(0,1,3,2).flip([3])
            if ifcorner:
                att = G.attractor(G.orient_max(out1[i],'xy'),kw,kd)
            else:
                att = G.attractor(G.orient_max(out1[i],'x' if i%2==0 else 'y'),kw,kd)
            out.append(att[0])
            out_weight.append(att[1])
            out_dist_x.append(att[2])
            out_dist_y.append(att[3])
        out = torch.cat(out,dim=1)
        out_weight = torch.cat(out_weight,dim=1)
        out_dist_x = torch.cat(out_dist_x,dim=1)
        out_dist_y = torch.cat(out_dist_y,dim=1)

        # out1 = torch.cat(out1,dim=1)
        return out,[out_weight,out_dist_x,out_dist_y]

class Rotate_BN(nn.Module):
    def __init__(self,channel,r,momentrm=0.1):
        super(Rotate_BN,self).__init__()
        self.bn = nn.BatchNorm2d(int(channel/r),momentum=0.1/r)
        self.r =r

    def forward(x):
        return G.BN(x,self.bn,r=self.r)

class Rotate_Conv(nn.Module):
    def __init__(self,in_channel,out_channel,r=4,kernel_size=(3,3),padding=(1,1),stride=1,device='cuda',sep=False,ifbn=True):
        super(Rotate_Conv,self).__init__()
        self.r = r
        self.ifbn=ifbn
        self.sep = sep
        self.conv = nn.Conv2d(in_channel, int(out_channel / self.r), kernel_size,bias=False)
        # if self.ifbn:
        self.bn = nn.BatchNorm2d(int(out_channel / self.r),momentum=0.1/self.r)
        self.padding = padding
        self.stride = stride

    def forward(self,x):
        # out = []
        w = []
        w1 = self.conv.weight
        for i in range(self.r):
            if i!=0:
                if not self.sep:
                    w1 = self.rotateChannel(w1,r=self.r)
                w1 = w1.permute(0,1,3,2).flip([3])
            w.append(w1)
        w = torch.cat(w,dim=0)
        out = F.conv2d(x,w,padding=self.padding,stride=self.stride)
        # out.append(F.conv2d(x,w1,padding=self.padding,stride=self.stride))
        #     # out[i] = G.orient_max(out[i],'x' if i%2==0 else 'y')
        # out = torch.cat(out,dim=1)
        if self.ifbn:
            out = G.BN(out, self.bn, r=self.r)
        else:
            b = self.bn.bias.reshape(1,-1,1,1)
            b = [b for _ in range(self.r)]
            out = out + torch.cat(b,dim=1)
        return out
    
    def rotateChannel(self,x, r):
        # print('rotate')
        a = x.size(1)
        # print(x.size())
        x = torch.split(x, int(a / r), dim=1)
        y = torch.cat(x[0:-1], dim=1)
        y = torch.cat((x[-1], y), dim=1)
        return y


class Nonpolar_Conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=(3,3),padding=(1,1),stride=1,device='cuda',ifbn=True):
        super(Nonpolar_Conv,self).__init__()
        self.r = 4
        self.ifbn = ifbn
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,bias=False)
        if self.ifbn:
            self.bn = nn.BatchNorm2d(out_channel,momentum=0.1)
        self.padding = padding
        self.stride = stride

    def forward(self,x):
        w = self.conv.weight
        for i in range(self.r):
            if i==0:
                w1 = w.permute(0,1,3,2).flip([3])
                w2 = w.permute(0,1,3,2).flip([3])
            else:
                w1 = w1.permute(0,1,3,2).flip([3])
                w2 += w1
        w1 /= self.r
        out = F.conv2d(x,w1,padding=self.padding,stride=self.stride)
        if self.ifbn:
            out = self.bn(out)
        return out

class Orient_Conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=(3,3),padding=(1,1),stride=1,device='cuda',ifbn=True):
        super(Orient_Conv,self).__init__()
        self.r = 4
        self.ifbn=ifbn
        self.conv = nn.Conv2d(int(in_channel / self.r), int(out_channel / self.r), kernel_size,bias=False)
        if self.ifbn:
            self.bn = nn.BatchNorm2d(int(out_channel / self.r),momentum=0.1/self.r)
        self.padding = padding
        self.stride = stride
        # self.wr = nn.Parameter((torch.Tensor(int(out_channel / self.r), int(in_channel / self.r),kernel_size[0],kernel_size[1]).uniform_(0,1)*np.pi).to(device),requires_grad=True)
        self.wr = (torch.Tensor(int(out_channel / self.r), int(in_channel / self.r),kernel_size[0],kernel_size[1]).uniform_(0,1)*np.pi).to(device)
        self.theta = torch.Tensor([np.pi/2]).to(device)

    def forward(self,x,r=None):
        out = []
        wr1 =torch.cat([self.wr-i*self.theta for i in range(self.r)],dim=1)
        # wr1 = (wr1%(2.*np.pi)).pow(2)
        f = x
        w1 = torch.cat([self.conv.weight for i in range(self.r)],dim=1)
        for i in range(self.r):
            if i!= 0:
                wr1 = wr1 + self.theta
                wr1 = wr1.permute(0,1,3,2).flip([3])
                w1 = w1.permute(0,1,3,2).flip([3])
            out.append(F.conv2d(f,w1*F.relu(wr1.cos()),padding=self.padding,stride=self.stride))
            # out.append(F.conv2d(f,w1*wr1,padding=self.padding,stride=self.stride))
            # out[i] = G.orient_max(out[i],'x' if i%2==0 else 'y')
        out = torch.cat(out,dim=1)
        if self.ifbn:
            out = G.BN(out,self.bn)
        return out

class Orient_Conv_out(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3,3), padding=(1,1), stride=1, device='cuda'):
        super(Orient_Conv_out, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,bias=False)
        self.eps = torch.Tensor([1e-5]).to(device)
        self.two = torch.Tensor([2.]).to(device)
        self.padding = padding
        self.stride = stride

    def orient(self,x):
        c = (x[0] - x[2])/self.two
        s = (x[1] - x[3])/self.two
        r = (c.pow(2) + s.pow(2)).sqrt()
        return r, [c/(r+self.eps),s/(r+self.eps)]

    def forward(self, x):
        out = []
        w1 = self.conv.weight
        for i in range(4):
            if i == 0:
                out.append(F.conv2d(x,w1,padding=self.padding,stride=self.stride))
            else:
                w1 = w1.permute(0,1,3,2).flip([3])
                out.append(F.conv2d(x,w1,padding=self.padding,stride=self.stride))
        out,r = self.orient(out)
        return out,r

class Orient_Conv_inout(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3,3), padding=(1,1,1,1),stride =1, device='cuda'):
        super(Orient_Conv_inout,self).__init__()
        b = 100.
        self.a=torch.Tensor([b]).to(device)
        self.kh = kernel_size[0]
        self.kw = kernel_size[1]
        self.pad = padding
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,bias=False)
        self.wr = Variable((torch.randn(out_channel,in_channel,self.kh,self.kw)*np.pi*b).to(device),requires_grad=False)
        self.orient = torch.Tensor([i*np.pi/2 for i in range(4)]).to(device)

    def R(self,c,s,x,y):
        return c*x-s*y,s*x+c*y

    def orient_mul(self,f,r,w,wr):
        lambda_theta = F.relu(wr.cos()*r[0]+wr.sin()*r[1])
        return (f*w*lambda_theta).sum(dim=2,keepdim=True)

    def conv_single(self,f,r,w,wr):
        # out = []
        he = f.size(3)-self.pad[0]-self.pad[1]
        we = f.size(4)-self.pad[2]-self.pad[3]
        cnt = 0
        s = self.stride
        # print('conv_single f',f.size(),'r',r[0].size(),r[1].size(),'w',w.size(),'wr',wr.size())
        # print(f[:,:,:,1:1+he:s,1:1+we:s].size())

        for i in range(self.kh):
            for j in range(self.kw):
                rr = [r[0][:,:,:,i:i+he:s,j:j+we:s],r[1][:,:,:,i:i+he:s,j:j+we:s]]
                if cnt == 0:
                    out = self.orient_mul(f[:,:,:,i:i+he:s,j:j+we:s],rr,w[:,:,:,i:i+1,j:j+1],wr[:,:,:,i:i+1,j:j+1])
                    cnt =1
                else:
                    out += self.orient_mul(f[:,:,:,i:i+he:s,j:j+we:s],rr,w[:,:,:,i:i+1,j:j+1],wr[:,:,:,i:i+1,j:j+1])
                    
        return out
    
    def forward(self,f,r):
        def orient_conv_max(weight,value,dim=2,ifshow=False):
            weight = torch.cat(weight,dim=dim)
            index = weight.argmax(dim=dim,keepdim=True)
            for i,a in enumerate(value):
                value[i] = torch.cat(value[i],dim=dim).gather(dim,index).squeeze(dim)
            weight = weight.gather(dim,index).squeeze(dim)

            return weight,value

        out = []
        out_r = []
        f = F.pad(f,self.pad).unsqueeze(1)
        r = [F.pad(r[0],self.pad).unsqueeze(1),F.pad(r[1],self.pad).unsqueeze(1)]
        w = self.conv.weight.unsqueeze(0)
        wr = (self.wr/self.a).unsqueeze(0)
        for i in range(4):
            if i==0:
                out.append(self.conv_single(f,r,w,wr))
            else:
                w = w.permute(0,1,2,4,3).flip([4])
                out.append(self.conv_single(f,r,w,wr+self.orient[i]))
            out_r.append(out[i] * 0. + self.orient[i])
        out,out_r = orient_conv_max(out,[out_r])

        return out,[out_r[0].cos(),out_r[0].sin()]

class Norm_Conv(nn.Module):
    def __init__(self, in_planes, planes,ifbn=True):
        super(Norm_Conv, self).__init__()
        self.ifbn = ifbn
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv(x)
        if self.ifbn:
            return self.bn(out)
        else:
            b = self.bn.bias.reshape(1,-1,1,1)
            return out+b

class MyLeNetNorm(nn.Module):
    def __init__(self):
        super(MyLeNetNorm,self).__init__()

        base = 16
        self.rconv1 = Norm_Conv(1,base)
        self.rconv3 = Norm_Conv(base,base)
        self.rconv5 = Norm_Conv(base,base)

        self.conv7 = nn.Conv2d(base, 10, 3)

        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()

    def forward(self,x):
        out = F.relu(self.rconv1(x))
        out = F.max_pool2d(out,2)
        out = F.relu(self.rconv3(out))
        out = F.max_pool2d(out,2)
        out = F.relu(self.rconv5(out))
        a = np.random.randint(0,2,(2))
        out = F.pad(out,(a[0],1-a[0],a[1],1-a[1]))
        out = F.max_pool2d(out,2)
        out = self.conv7(out)

        out = F.max_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)

        return out,0.

    def observe(self,x):
        return 0


class MyLeNetRotateInvariantNew_nms(nn.Module):
    def __init__(self,logger=None):
        super(MyLeNetRotateInvariantNew_nms, self).__init__()
        self.r = 4
        base = 40

        self.rconv1 = Rotate_Conv(1,base,kernel_size=(5,5),padding=(2,2),sep=True)
        self.rconv2 = Rotate_Conv(base,base)
        self.rconv3 = Rotate_Conv(base,base)
        self.rconv4 = Rotate_Conv(base,base)
        self.rconv5 = Rotate_Conv(base,base)
        self.rconv6 = Rotate_Conv(base,base)
        self.conv7 = Rotate_Conv(base, 10*self.r, ifbn=False)

        # self.rconv3 = Orient_Conv(base,base)
        # self.rconv5 = Orient_Conv(base,base)
        # self.conv7 = Orient_Conv(base, 10*self.r, ifbn=False)

        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()


    def forward(self,x):

        # x = F.pad(x,(2,2,2,2))
        out = F.relu(self.rconv1(x))
        out = F.relu(self.rconv2(out))
        out = F.max_pool2d(out,2)
        out = F.relu(self.rconv3(out))
        out = F.relu(self.rconv4(out))
        # out = F.max_pool2d(out,2)
        out = F.relu(self.rconv5(out))
        out = F.relu(self.rconv6(out))
        # a = np.random.randint(0,2,(2))
        # out = F.pad(out,(a[0],1-a[0],a[1],1-a[1]))
        # out = F.max_pool2d(out,2)
        out = self.conv7(out)

        out = self.rotatedim_max(out)
        out = F.max_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)

        return out,0

    def observe(self,x):
        x = F.pad(x,(2,2,2,2))
        out1 = F.relu(self.rconv1(x))
        out1 = F.max_pool2d(out1,2)
        out1 = F.relu(self.rconv3(out1))
        out1 = F.max_pool2d(out1,2)
        out1 = F.relu(self.rconv5(out1))
        # out1 = F.pad(out1,(1,0,1,0))
        out1 = F.max_pool2d(out1,2)
        out1 = self.conv7(out1)
        out1 = self.rotatedim_max(out1)

        # x = x.permute(0,1,3,2).flip([3])
        # out2 = F.relu(self.rconv1(x))
        # out2 = F.max_pool2d(out2,2)
        # out2 = F.relu(self.rconv3(out2))
        # out2 = F.relu(self.rconv5(out2))
        # out2 = F.max_pool2d(out2,2)
        # out2 = self.conv7(out2)
        # out2 = self.rotatedim_max(out2)

        self.show_multi(out1[0,1::5,:,:])
        # self.show_multi(out2[0,1::5,:,:])

        return

    def nonpolar_add(self,rot,non):
        a = rot.size(1)
        # print(x.size())
        rt = torch.split(rot, int(a / self.r), dim=1)
        rt = [rt[i]+non for i in range(self.r)]
        return torch.cat(rt,dim=1)

    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, self.r, int(s1 / self.r), s2, s3)
        y = torch.max(x, dim=1, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=1)
        y = y.ne(x).float()
        x = x * y
        return x.mean()

    def rotatedim_sum(self,x):
        s0, s1, s2, s3 = x.size()
        return x.view(s0, self.r, int(s1 / self.r), s2, s3).sum(dim=1)

    def rotatedim_max(self,x):
        s0,s1,s2,s3 = x.size()
        return x.view(s0,self.r,int(s1/self.r),s2,s3).max(dim=1)[0]

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
            b = b-np.min(b,axis=(1,2),keepdims=True)
        s1 = x.size(0)
        for i in range(s1):
            self.viz.image(b[i]/np.max(b))
        if ifsum:
            b = np.sum(b,axis=0)
            self.viz.image(b/np.max(b))
    
