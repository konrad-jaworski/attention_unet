import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Class which defines main convblock used in the Unet down and up sampling stages
    
    """
    def __init__(self, in_channels, out_channels,kernel_size=(3,3,1),stride=(1,1,1),padding=(1,1,0)):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding,stride=stride), # in this step we only change number of filters
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding,stride=stride), # This step does notchange even number of filters
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)
    

class DownBlock(nn.Module):
    """
    Downsampling block which halfs the size of it input in HxW 
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.convblock=ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,1)) # Reduction of spatial dimmensions by half

    def forward(self, x):
        x = self.convblock(x)
        x_pooled = self.maxpool(x)
        return x_pooled, x # Saved for concatenation
    

class UpBlock(nn.Module):
    """
    Uppsampling block which increase size of the input by factor of 2 in HxW
    """
    def __init__(self,in_channels,out_channels):
        super(UpBlock,self).__init__()

        self.convblock=ConvBlock(in_channels=in_channels,out_channels=out_channels)
        self.up=nn.ConvTranspose3d(in_channels,out_channels,kernel_size=(4,4,1),stride=(2,2,1),padding=(1,1,0)) # We have learnable upsample of the data, we upsample it to twice the size

    def forward(self,x_up,x_skip):
        x_up=self.up(x_up)
        x=torch.cat([x_up,x_skip],dim=1)
        return self.convblock(x)
    
class GatingGridSignal(nn.Module):
    """
    Gating signal placed at the bottleneck of the Unet
    """
    def __init__(self,in_size, out_size, kernel_size=(1,1,1)):
        super(GatingGridSignal,self).__init__()

        # We are using this block for the purpose changing number of filters
        self.conv1=nn.Sequential(nn.Conv3d(in_size,out_size,kernel_size,(1,1,1),(0,0,0)), # Gating signal only transform number of channels
                                 nn.BatchNorm3d(out_size),
                                 nn.ReLU(inplace=True))
        
    def forward(self,inputs):
        outputs=self.conv1(inputs)
        return outputs
    
class AttentionGate(nn.Module):
    """
    Attention fusion part of attention network
    """
    def __init__(self,in_channels,gating_channels,inter_channels,sub_sample_factor=(2,2,2)):
        super(AttentionGate, self).__init__()
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        # Subsampling for feature map to match gating signal
        self.sub_sample_factor = sub_sample_factor


        self.W=nn.Sequential(nn.Conv3d(in_channels=self.in_channels,
                                       out_channels=self.in_channels,
                                       kernel_size=1,stride=1,padding=0), # This convolution does not change shape of its input but provide some linear transformation
                                       nn.BatchNorm3d(self.in_channels))

        self.Wg=nn.Conv3d(in_channels=self.gating_channels,
                          out_channels=self.inter_channels,
                          kernel_size=1,
                          stride=1,
                          bias=True,
                          padding=0) # This convolution only change number of filters used

        self.Wx=nn.Conv3d(in_channels=self.in_channels,
                          out_channels=self.inter_channels,
                          kernel_size=self.sub_sample_factor,
                          stride=self.sub_sample_factor,
                          bias=False,
                          padding=0) # This convolution layer halfs all of the dimension of its input and convert number of channels to internal number of channels

        self.Psi=nn.Conv3d(in_channels=self.inter_channels,
                           out_channels=1,
                           kernel_size=1,
                           stride=1,
                           bias=True,
                           padding=0) # This convolution convert number of channels to 1 but does not change other size of the input

    def attention_fusion(self,x,g):
        input_size=x.size()

        theta_x=self.Wx(x)
        theta_x_size=theta_x.size()

        phi_g=F.interpolate(self.Wg(g),size=theta_x_size[2:],mode='trilinear') # We interpolate the dimmension of the input to match skip connection
        f=F.relu(theta_x+phi_g,inplace=True)

        sigm_psi_f=F.sigmoid(self.Psi(f))
        sigm_psi_f=F.interpolate(sigm_psi_f,size=input_size[2:],mode='trilinear') # Our attention map
        y=sigm_psi_f.expand_as(x)*x # at this stage we multiply the input with attention map
        W_y=self.W(y)

        return W_y, sigm_psi_f

    def forward(self, x,g):
        output=self.attention_fusion(x,g)
        return output