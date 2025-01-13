from functools import partial
from timm.models.helpers import named_apply

from timm.models.layers import trunc_normal_
import math
from .Res2Net import res2net50_v1b_26w_4s,res2net101_v1b_26w_4s
from .UNet_v2 import BasicConv2d,SDI
from .conv_dnm import *

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class LAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RRM(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        x = x + idn
        x = F.gelu(x.real)
        return x

class DNM_Conv(nn.Module):
    def __init__(self, input_size, out_size, M, activation=F.relu):
        super(DNM_Conv, self).__init__()#
        DNM_W = torch.rand([out_size, M, input_size])  #
        DNM_q = torch.rand([out_size, M, input_size])
        qs = torch.rand(1)
        qs = torch.tensor(qs).to(DEVICE)
        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(DNM_q)})
        self.qs = qs
        self.activation = activation

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)

    def forward(self, x):
        out_size, M, _ = self.params['DNM_W'].shape
        x = torch.permute(x, (0, 2, 3, 1))  #
        x = torch.unsqueeze(x, 3) #
        x = torch.unsqueeze(x, 4) #
        x = self.norm1(x)  # norm

        x = x.repeat(1, 1, 1, out_size, M, 1) #
        x = F.relu(torch.mul(x, self.params['DNM_W']) - self.params['q'])

        x = self.norm2(x)  # norm、

        x = torch.sum(x, 5)
        x = torch.sum(x, 4)
        x = torch.permute(x, (0, 3, 1, 2))

        # Soma 胞体层
        if self.activation != None:
            # x = self.activation(self.k * (x - self.qs))
            x = self.activation(x - self.qs)
        return x

class conv_dnm(nn.Module):
    def __init__(self, in_channal=64, out_channal=1, m=10):
        super(conv_dnm, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.Dnm_conv2d = DNM_Conv(in_channal, out_channal, m, activation=None)
        # self.conv2d = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.Dnm_conv2d(x)
        return x

class BFUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()
        self.c_list = c_list
        self.bridge = bridge
        self.gt_ds = gt_ds

        if gt_ds:
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(c_list[5], c_list[4], kernel_size=3, stride=1, padding=1,
                               bias=False)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(c_list[4], c_list[3], kernel_size=4, stride=2, padding=1,
                               bias=False)
        ) 
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(c_list[3], c_list[2], kernel_size=4, stride=2, padding=1,
                               bias=False)
        )  
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(c_list[2], c_list[1], kernel_size=4, stride=2, padding=1,
                               bias=False)
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])


        self.apply(self._init_weights)

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        LAG_ks = 3
        self.LAG4 = LAG(F_g= c_list[3], F_l= c_list[3], F_int= c_list[3]//2, kernel_size=LAG_ks, groups= c_list[3]//2, activation='relu')
        self.LAG3 = LAG(F_g= c_list[2], F_l= c_list[2], F_int= c_list[2]//2, kernel_size=LAG_ks, groups= c_list[2]//2, activation='relu')
        self.LAG2 = LAG(F_g= c_list[1], F_l= c_list[1], F_int= c_list[1]//2, kernel_size=LAG_ks, groups= c_list[1]//2, activation='relu')

        channel = 32
        self.Translayer_1 = BasicConv2d(256, c_list[1], 1)
        self.Translayer_1_fft = BasicConv2d(256, c_list[1], 1)
        self.Translayer_2 = BasicConv2d(512, c_list[2], 1)
        self.Translayer_2_fft = BasicConv2d(512, c_list[2], 1)
        self.Translayer_3 = BasicConv2d(1024, c_list[3], 1)
        self.Translayer_3_fft = BasicConv2d(1024, c_list[3], 1)
        self.Translayer_4 = BasicConv2d(2048, c_list[4], 1)
        self.Translayer_4_fft = BasicConv2d(2048, c_list[4], 1)

        base_dims = 64
        hidden_dim = int(base_dims // 4)

        dims = [64, 128, 256, 512]
        depths = [2, 2, 9, 2]
        norm_layer = nn.LayerNorm

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.num_layers = len(depths)
        self.embed_dim = dims[0]

        self.norm_up = norm_layer(64)

        self.rrm1 = RRM(c_list[1])
        self.rrm2 = RRM(c_list[2])
        self.rrm3 = RRM(c_list[3])
        self.rrm4 = RRM(c_list[4])

        self.rrm5 = RRM(c_list[3])
        self.rrm6 = RRM(c_list[2])
        self.rrm7 = RRM(c_list[1])


        self.Dnm_conv2d = conv_dnm(c_list[0], 1, 10)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        t_f1,t_f2,t_f3,t_f4 = self.resnet(x)

        f1 = self.Translayer_1(t_f1)
        f1 = self.rrm1(f1)

        f2 = self.Translayer_2(t_f2)
        f2 = self.rrm2(f2)

        f3 = self.Translayer_3(t_f3) 
        f3 = self.rrm3(f3)

        f4 = self.Translayer_4(t_f4)
        f4 = self.rrm4(f4)

        t2=f1
        t3=f2
        t4=f3
        t5=f4

        #----------------------------decoder--------------------------------------------------------------------
        out5 = t5
        out4 = F.gelu(self.dbn2(self.decoder2(out5)))  
        out4 = self.rrm5(out4)

        t4 = self.LAG4(g=out4, x=t4)
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        out4 = torch.add(out4, t4) 

        out3 = F.gelu(self.dbn3(self.decoder3(out4)))
        out3 = self.rrm6(out3)
        t3 = self.LAG3(g=out3, x=t3)
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        out3 = torch.add(out3, t3) 


        out2 = F.gelu(self.dbn4(self.decoder4(out3)))
        out2 = self.rrm7(out2)
        t2 = self.LAG2(g=out2, x=t2)
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        out2 = torch.add(out2, t2) 
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',align_corners=True))  # b, c0, H/2, W/2

        gt_pre1 = self.gt_conv5(out1)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        out1 = self.Dnm_conv2d(out1)
        out0 = F.interpolate(out1,scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W

        if self.gt_ds:
            return (torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)
