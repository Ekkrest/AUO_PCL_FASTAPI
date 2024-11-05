import torch
import torch.nn as  nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

BATCH_NORM_EPSILON = 1e-5

class ConvBlockOnly(nn.Module):
    def __init__(self, In_Channels, Out_Channels, Kernel_Size, Stride, Padding):
        super(ConvBlockOnly, self).__init__()
        self.Conv = nn.Conv2d(in_channels=In_Channels, out_channels=Out_Channels, kernel_size=Kernel_Size, stride=Stride, padding=Padding, bias=False)

    def forward(self, x):
        x = self.Conv(x)
     
        return x

class BatchNorm(nn.Module):
    def __init__(self, Out_Channels):
        super(BatchNorm, self).__init__()
        self.Batch_Norm = nn.BatchNorm2d(Out_Channels)
        self.Activ_Func = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Batch_Norm(x)
        x = self.Activ_Func(x)
        
        return x

class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))

        return out

class ChannelAttentionBlock(nn.Module):
    def __init__(self, In_Channels, Out_Channels):
        super(ChannelAttentionBlock, self).__init__()

        self.Conv = nn.Conv2d(in_channels=In_Channels, out_channels=Out_Channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Glob_Avg_Pool = nn.AdaptiveAvgPool2d(1)
        self.Activ_Func = nn.Sigmoid()

    def forward(self, x): 
        x = self.Conv(x)
        x = self.Glob_Avg_Pool(x)
        x = self.Activ_Func(x)

        return x 

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal', shrink=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes // 2, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion // 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion // 2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

        self.block1 = nn.Sequential(
            ConvBlockOnly(In_Channels=(planes * self.expansion) // 2, Out_Channels=shrink, Kernel_Size=1, Stride=1, Padding=0)
        )
        
        # Second Block contains filters with kernel size 3x3 
        self.block2 = nn.Sequential(
            ConvBlockOnly(In_Channels=(planes * self.expansion) // 2, Out_Channels=shrink, Kernel_Size=3, Stride=1, Padding=1),
        )
        
        # Third Block same as second block unless we'll replace the 3x3 filter with 5x5 
        self.block3 = nn.Sequential(
            ConvBlockOnly(In_Channels=(planes * self.expansion) // 2, Out_Channels=shrink, Kernel_Size=5, Stride=1, Padding=2),
        )

        self.block4 = nn.Sequential(
            ChannelAttentionBlock(In_Channels=shrink * 6, Out_Channels=planes * self.expansion)
        )

        self.SpatialBlock = nn.Sequential(
            SpatialAttentionBlock(in_channels=2, out_channels=1)
        )

        self.BNRL = nn.Sequential(
            BatchNorm(shrink * 6)
        )

    def forward(self, x):
        residual = x

        split_position = x.size(1) // 2
        a = x[:, :split_position, :, :]
        b = x[:, split_position:, :, :]

        a = self.relu(self.bn1(self.conv1(a)))
        b = self.relu(self.bn1(self.conv1(b)))

        spx_a = torch.split(a, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp_a = spx_a[i]
            else:
                sp_a = sp_a + spx_a[i]
            sp_a = self.convs[i](sp_a)
            sp_a = self.relu(self.bns[i](sp_a))
            if i==0:
                a = sp_a
            else:
                a = torch.cat((a, sp_a), 1)
        if self.scale != 1 and self.stype=='normal':
            a = torch.cat((a, spx_a[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
            a = torch.cat((a, self.pool(spx_a[self.nums])),1)

        a = self.conv3(a)
        a = self.bn3(a)

        spx_b = torch.split(b, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp_b = spx_b[i]
            else:
                sp_b = sp_b + spx_b[i]
            sp_b = self.convs[i](sp_b)
            sp_b = self.relu(self.bns[i](sp_b))
            if i==0:
                b = sp_b
            else:
                b = torch.cat((b, sp_b), 1)
        if self.scale != 1 and self.stype=='normal':
            b = torch.cat((b, spx_b[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
            b = torch.cat((b, self.pool(spx_b[self.nums])),1)           
        b = self.conv3(b)
        b = self.bn3(b)

        first_block_out_a = self.block1(a)
        second_block_out_a = self.block2(a)
        third_block_out_a = self.block3(a)

        first_block_out_b = self.block1(b)
        second_block_out_b = self.block2(b)
        third_block_out_b = self.block3(b)

        concat_1x1 = torch.cat([first_block_out_a, first_block_out_b], dim=1)
        concat_3x3 = torch.cat([second_block_out_a, second_block_out_b], dim=1)
        concat_5x5 = torch.cat([third_block_out_a, third_block_out_b], dim=1)

        concatenated_Outs = torch.cat([concat_1x1, concat_3x3, concat_5x5], dim=1)

        bn_output = self.BNRL(concatenated_Outs)
        channelAttention = self.block4(bn_output)

        out = torch.cat((a, b), dim=1)
        out = channelAttention * out

        normal_features = out[:, :split_position, :, :]
        defect_features = out[:, split_position:, :, :]

        spatialAttention = self.SpatialBlock(defect_features)
        normal_features = spatialAttention * normal_features

        out = torch.cat((normal_features, defect_features), dim=1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out    

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=128):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        shrink = 2048 // (planes * block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale, shrink=shrink))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale, shrink=shrink))

        return nn.Sequential(*layers)

    def forward(self, x):
        split_position = x.size(1) // 2
        img1 = x[:, :split_position, :, :] # original image
        img2 = x[:, split_position:, :, :] # position image
        conv1_result = self.relu(self.bn1(self.conv1(img1)))
        conv2_result = self.relu(self.bn1(self.conv1(img2)))

        x = torch.cat((conv1_result, conv2_result), dim=1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)      
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x        

class ContrastiveHead(nn.Module):
    def __init__(self, channels_in, out_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                dim, relu = channels_in, True
            else:
                dim, relu = out_dim, False
            self.layers.append(nn.Linear(channels_in, dim, bias=False))
            bn = nn.BatchNorm1d(dim, eps=BATCH_NORM_EPSILON, affine=True)
            if i == num_layers - 1:
                nn.init.zeros_(bn.bias)
            self.layers.append(bn)
            if relu:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x

class MocoV2_PCL(nn.Module):
    def __init__(self, model, head):
        super(MocoV2_PCL, self).__init__()
        
        self.encoder = model
        self.contrastive_head = head

    def forward(self, x):
        x = self.encoder(x)
        x = self.contrastive_head(x)
        return x  