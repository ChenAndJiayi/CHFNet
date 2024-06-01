
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from .mit import mit_b4
import numpy as np

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=1, stride=1,
                              padding=0)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, affine=True, eps=1.1e-5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                 padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        module.append(nn.Conv2d(in_channels = in_ch,out_channels=out_ch, kernel_size=kSize, stride=stride,
                           padding=padding, dilation=dilation, groups=1, bias=bias))
        # decide use GN or BN
        if norm == 'GN':
            module.append(nn.GroupNorm(
                num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(out_ch, eps=0.001,
                          momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)

        self.module = nn.Sequential(*module)

    def forward(self, x):
        out = self.module(x)
        return out

    
class LightDASP(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(LightDASP, self).__init__()

        self.reduction1 = nn.Conv2d(in_feat, in_feat//2, kernel_size=1, stride=1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(myConv(in_feat//2, in_feat//4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act, num_groups=(in_feat//2)),
                                     myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act, num_groups=(in_feat//4)))
        self.aspp_d6 = nn.Sequential(myConv(in_feat//2 + in_feat//4, in_feat//4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act, num_groups=( in_feat//4)),
                                     myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act, num_groups=(in_feat//4)))
        self.aspp_d12 = nn.Sequential(myConv(in_feat, in_feat//4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act, num_groups=(in_feat//4)),
                                      myConv(in_feat//4, in_feat//4, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm, act=act, num_groups=(in_feat//4)))
        self.reduction2 = myConv(((in_feat//4)*3) + (in_feat//2), in_feat, kSize=3, stride=1,padding=1, bias=False, norm=norm, act=act, num_groups= in_feat)
    
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        out = self.reduction2(torch.cat([x, d3, d6, d12], dim=1))
        return out

class CHF_decoder_all(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        # 第5层block，最后一层没有bn和relu，因为这个已经算是output了，但是我看别人的代码还是有BN和relu的这个后续再说吧
        self.ASPP5 = LightDASP('BN', 'ReLU', 128)
        self.conv_block5 = nn.Sequential(
                                         conv3x3(128, 64),
                                         conv3x3(64, 32),
                                         nn.Conv2d(32, 1,3,1,1) 
                                         )



        # 第4层 block
        self.conv4_up = conv1x1(128, 320) # 只有一个卷积层, 768-->384
        self.ASPP4 = LightDASP('BN', 'ReLU', 160)      # aspp4 channel is 640
        self.conv_block4 = nn.Sequential(
                                         conv3x3(164, 64),
                                         conv3x3(64, 32),
                                         nn.Conv2d(32, 1,3,1,1)
                                         )
        
        # 第3层block
        self.conv3_up = nn.Sequential(conv3x3(288, 256))
        self.ASPP3 = LightDASP('BN', 'ReLU', 64 )
        self.conv_block3 = nn.Sequential(
                                         conv3x3(68, 32),
                                         nn.Conv2d(32, 1,3,1,1)
                                         )

        # 第2层block
        self.ASPP2 = LightDASP('BN', 'ReLU', 32 )
        self.conv_block2 = nn.Sequential(
                                         conv3x3(36, 16),
                                         nn.Conv2d(16, 1,3,1,1)
                                         )
        
        # 第1层block
        self.ASPP1 = LightDASP('BN', 'ReLU', 8 )
        self.conv_block1 = nn.Sequential(conv3x3(12, 8),conv3x3(8, 4),nn.Conv2d(4, 1,3,1,1))
        self.up_2= nn.PixelShuffle(2)
        
        #dimList = [64, 128, 320, 512], act = 'ReLU', norm = 'BN', self.ASPP = Dilated_bottleNeck(norm, act, dimList)

    def forward(self, x_1, x_2, x_3, x_4, lap_list):

        rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = lap_list[0], lap_list[1], lap_list[2], lap_list[3]

        aspp5_in    = self.up_2(x_4)                # 512/4=128,1/16
        aspp5       = self.ASPP5(aspp5_in)          # 128
        att5        = self.conv_block5(aspp5)       # 1/16
        att5_up     = F.interpolate(att5, scale_factor=2, mode='bilinear', align_corners=True)  # 1/8

        conv4       = self.conv4_up(aspp5)              # 128-->320
        aspp4_in    = torch.cat([x_3 , conv4],dim=1)    # 640
        sizeup_4    = self.up_2(aspp4_in)               # 640/4 = 160, 1/8
        aspp4       = self.ASPP4(sizeup_4)              # 160
        aspp4_cat   = torch.cat([aspp4 ,rgb_lv4,att5_up ],dim=1) # 160 + 3 + 1, 1/8
        att4        = self.conv_block4(aspp4_cat)  #  这里加上rgb_lv4的边缘图
        att4_up     = F.interpolate(att4, scale_factor=2, mode='bilinear', align_corners=True) # 1/4

        conv3       = self.conv3_up(torch.cat([x_2,aspp4],dim=1)) # 128+160-->256
        sizeup_3    = self.up_2(conv3) #256/4=64,,1/4
        aspp3       = self.ASPP3(sizeup_3)              # 64,1/4
        aspp3_cat   = torch.cat([aspp3,rgb_lv3,att4_up],dim=1) # 64+3+1, 1/4
        att3        = self.conv_block3(aspp3_cat)  # 68
        att3_up     = F.interpolate(att3, scale_factor=2, mode='bilinear', align_corners=True)  # 1/2

        aspp2_in    = torch.cat([x_1,aspp3],dim=1) # 64+64=128
        sizeup_2    = self.up_2(aspp2_in) # 128/4=32, 1/2
        aspp2       = self.ASPP2(sizeup_2) # 32
        aspp2_cat   = torch.cat([aspp2 ,rgb_lv2,att3_up ],dim=1) # 32+3+1=20
        att2        = self.conv_block2(aspp2_cat)  # 20-->1
        att2_up     = F.interpolate(att2, scale_factor=2, mode='bilinear', align_corners=True)  # 1/1
        
        sizeup_1    = self.up_2(aspp2)  # 8,1/1
        aspp1       = self.ASPP1(sizeup_1) # 8
        att_out     = self.conv_block1(torch.cat([aspp1 ,rgb_lv1,att2_up],dim=1))   # 8+3+1, 1/1

        return att5_up, att4_up, att3_up, att2_up, att_out


class CHFNet(nn.Module):
    def __init__(self, max_depth=10.0, is_train=True):
        super().__init__()
        self.max_depth = max_depth
        self.encoder = mit_b4()
        if is_train:
            ckpt_path = './models/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './code/models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        self.decoder = CHF_decoder_all()
        #
        num_params = sum([np.prod(p.size()) for p in self.encoder.parameters()])
        print("Total encoder number of parameters: {}".format(num_params))
        num_params = sum([np.prod(p.size()) for p in self.decoder.parameters()])
        print("Total decoder number of parameters: {}".format(num_params))
        #
    def forward(self, x):
        conv1, conv2, conv3, conv4 = self.encoder(x)

        rgb_down2   = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down4   = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down8   = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down16  = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear', align_corners=True)
        rgb_down32  = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear', align_corners=True)

        rgb_up16    = F.interpolate(rgb_down32, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up8     = F.interpolate(rgb_down16, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up4     = F.interpolate(rgb_down8, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up2     = F.interpolate(rgb_down4, scale_factor=2, mode='bilinear', align_corners=True)
        rgb_up      = F.interpolate(rgb_down2, scale_factor=2, mode='bilinear', align_corners=True)

        lap1        = x - rgb_up
        lap2        = rgb_down2 - rgb_up2
        lap3        = rgb_down4 - rgb_up4
        lap4        = rgb_down8 - rgb_up8
        lap5        = rgb_down16 - rgb_up16
        lap_list    = [ lap4, lap3, lap2, lap1]

        att5, att4, att3, att2, att_out = self.decoder(conv1, conv2, conv3, conv4, lap_list)

        depth4 = att5
        depth3 = att4+0.1*F.interpolate(depth4, scale_factor=2, mode='bilinear', align_corners=True)
        depth2 = att3+0.2*F.interpolate(depth3, scale_factor=2, mode='bilinear', align_corners=True)
        depth1 = att2+0.3*F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)

        att_img= att_out+0.4*depth1
        out_depth = torch.sigmoid(att_img) * self.max_depth

        depth4 = F.interpolate(depth4, scale_factor=8, mode='bilinear', align_corners=True)
        depth4 = torch.sigmoid(depth4) * self.max_depth
        depth3 = F.interpolate(depth3, scale_factor=4, mode='bilinear', align_corners=True)
        depth3 = torch.sigmoid(depth3) * self.max_depth
        depth2 = F.interpolate(depth2, scale_factor=2, mode='bilinear', align_corners=True)
        depth2 = torch.sigmoid(depth2) * self.max_depth
        depth1 = F.interpolate(depth1, scale_factor=2, mode='bilinear', align_corners=True)
        depth1 = torch.sigmoid(depth1) * self.max_depth

        return out_depth, depth4, depth3, depth2, depth1