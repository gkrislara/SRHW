import torch
import torch.nn as nn

class SRHW(nn.Module):
    def __init__(self,upscale=2,quant=False):
        super(SRHW,self).__init__()
        self.Conv1=nn.Conv2d(1,32,3,padding=(1,1),bias=False)
        nn.init.uniform_(self.Conv1.weight)
        self.DWConv1=nn.Conv2d(32,32,(1,5),padding=(0,2),groups=32,bias=False)
        nn.init.uniform_(self.DWConv1.weight)
        self.PWConv1=nn.Conv2d(32,16,1,bias=False)
        nn.init.uniform_(self.PWConv1.weight)
        self.DWConv2=nn.Conv2d(16,16,(1,5),padding=(0,2),groups=16,bias=False)
        nn.init.uniform_(self.DWConv2.weight)
        self.PWConv2=nn.Conv2d(16,32,1,bias=False)
        nn.init.uniform_(self.PWConv2.weight)
        self.DWConv3=nn.Conv2d(32,32,3,padding=(1,1),groups=32,bias=False)
        nn.init.uniform_(self.DWConv3.weight)
        self.PWConv3=nn.Conv2d(32,16,1,bias=False)
        nn.init.uniform_(self.PWConv3.weight)
        self.DWConv4=nn.Conv2d(16,16,3,padding=(1,1),groups=16,bias=False)
        nn.init.uniform_(self.DWConv4.weight)
        self.PWConv4=nn.Conv2d(16,upscale**2,1,bias=False)
        nn.init.uniform_(self.PWConv4.weight)
        self.PS=nn.PixelShuffle(upscale)
        self.relu=nn.ReLU(inplace=True)
        self.quant=quant

    def forward(self,x):
        x=self.Conv1(x)
        res=self.relu(x)
        res=self.relu(self.PWConv1(self.DWConv1(res)))
        res=self.PWConv2(self.DWConv2(res))
        x=x+res
        x=self.relu(x)
        x=self.relu(self.PWConv3(self.DWConv3(x)))
        x=self.PWConv4(self.DWConv4(x))
        if self.quant:
            return x
        else:
            x=self.PS(x)
            return x

