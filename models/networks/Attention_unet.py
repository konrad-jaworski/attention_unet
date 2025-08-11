import torch
from torch import nn
import torch.nn.functional as F
from models.layers.grid_attention import *

    
class AttentionUnet(nn.Module):
    """
    Attention Unet model
    """
    def __init__(self,n_classes,in_channels,attention_dsample=(2,2,2)):
        super(AttentionUnet,self).__init__()
        self.n_classes=n_classes
        self.in_channels=in_channels
        self.attention_dsample=attention_dsample
        filter_list=[64,128,256,512,1024]

        # Downsampling
        self.convdown1=DownBlock(in_channels=self.in_channels,out_channels=filter_list[0]) # Channels: 1 -> 64, Dimmensions: 512,512,301 -> 256,256,301
        self.convdown2=DownBlock(in_channels=filter_list[0],out_channels=filter_list[1]) # Channels: 64 -> 128, Dimmensions: 256,256,301 -> 128,128,301
        self.convdown3=DownBlock(in_channels=filter_list[1],out_channels=filter_list[2]) # Channels: 128 -> 256, Dimmensions: 128,128,301 -> 64,64,301
        self.convdown4=DownBlock(in_channels=filter_list[2],out_channels=filter_list[3]) # Channels: 256 -> 512, Dimmensions:  64,64,301 -> 32,32,301

        # Center and gating signal
        self.center=DownBlock(in_channels=filter_list[3],out_channels=filter_list[4]) # Channels: 512 -> 1024, Dimmensions: 32,32,301 -> 32,32,301 // We ommit maxpooling
        self.gating=GatingGridSignal(in_size=filter_list[4],out_size=filter_list[3]) # Channels: 1024 -> 512, Dimmensions:  32,32,301 -> 32,32,301 

        # attention blocks
        self.attention2=AttentionGate(in_channels=filter_list[1],gating_channels=filter_list[3],
                                      inter_channels=filter_list[1],sub_sample_factor=self.attention_dsample)
        
        self.attention3=AttentionGate(in_channels=filter_list[2],gating_channels=filter_list[3],
                                      inter_channels=filter_list[2],sub_sample_factor=self.attention_dsample)
        
        self.attention4=AttentionGate(in_channels=filter_list[3],gating_channels=filter_list[3],
                                      inter_channels=filter_list[3],sub_sample_factor=self.attention_dsample)
        
        # Upsampling
        self.convup4=UpBlock(in_channels=filter_list[4],out_channels=filter_list[3]) # Channels: 1024 -> 512, Dimmensions:  32,32,301 -> 64,64,301
        self.convup3=UpBlock(in_channels=filter_list[3],out_channels=filter_list[2]) # Channels: 512->256, Dimmensions:  64,64,301 -> 128,128,301
        self.convup2=UpBlock(in_channels=filter_list[2],out_channels=filter_list[1]) # Channels: 256->128, Dimmensions:  128,128,301 -> 256,256,301
        self.convup1=UpBlock(in_channels=filter_list[1],out_channels=filter_list[0]) # Channels: 128->64, Dimmensions:  256,256,301 -> 512,512,301

        # final conv for making dense prediction
        self.final=nn.Conv3d(1,n_classes,1)

    def forward(self,inputs):
        # Feature extraction
        conv1,x_1_skip=self.convdown1(inputs)

        conv2,x_2_skip=self.convdown2(conv1)

        conv3,x_3_skip=self.convdown3(conv2)

        conv4,x_4_skip=self.convdown4(conv3)

        # Gating signal generator
        _,center_skip=self.center(conv4)
        gating=self.gating(center_skip)

        # Attention Mechanism
        g_conv4,att4=self.attention4(x_4_skip,gating)
        g_conv3,att3=self.attention3(x_3_skip,gating)
        g_conv2,att2=self.attention2(x_2_skip,gating)

        # Upscaling part
        up4=self.convup4(center_skip,g_conv4)
        up3=self.convup3(up4,g_conv3)
        up2=self.convup2(up3,g_conv2)
        up1=self.convup1(up2,x_1_skip)

        final=self.final(up1)
        return final,att4,att3,att2