import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.HVI_transform import RGB_HVI
from basicsr.models.archs.arch_model import (
    LayerNorm2d, SimpleGate, Branch, FreMLP, DBlock,EBlock,
    SpatialBlock, FreqBlock, SimpleInteraction)


class CustomSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ACSN(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations=[1, 4, 9], 
                 extra_depth_wise=True):
        super(ACSN, self).__init__()
        
        self.hvi_transform = RGB_HVI()
        
        self.intro = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        
        self.padder_size = 2 ** len(self.encoders)        
        
        self.side_out = nn.Conv2d(in_channels=width * 2**len(self.encoders), out_channels=3, 
                                  kernel_size=3, stride=1, padding=1)
        
    def forward(self, input, side_loss=True, use_adapter=None):

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        
        x_hvi = self.hvi_transform.HVIT(input)
        
        x = self.intro(x_hvi)
        
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side_hvi = self.side_out(x_light)
        
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + x_hvi
        out = self.hvi_transform.PHVIT(x)
        out = out[:, :, :H, :W]  
        
        if side_loss:
            h_side, w_side = out_side_hvi.shape[2], out_side_hvi.shape[3]
            x_hvi_side = F.interpolate(x_hvi, size=(h_side, w_side), mode='bilinear', align_corners=False)
            out_side = self.hvi_transform.PHVIT(out_side_hvi + x_hvi_side)
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x      


if __name__ == '__main__':

    img_channel = 3
    width = 32
    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net = ACSN(img_channel=img_channel, 
                 width=width, 
                 middle_blk_num_enc=middle_blk_num_enc,
                 middle_blk_num_dec=middle_blk_num_dec,
                 enc_blk_nums=enc_blks, 
                 dec_blk_nums=dec_blks,
                 dilations=dilations,
                 extra_depth_wise=extra_depth_wise)
    
    test_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out_side, out = net(test_input, side_loss=True)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Side output shape: {out_side.shape}")
    
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (3, 256, 256), verbose=False, print_per_layer_stat=False)
    print(f"MACs: {macs}, Params: {params}")
