import torch
import torch.nn as nn
import torch.nn.functional as F

class Correlation(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, in1, in2):
        b, c, h, w = in1.shape
        max_d = self.max_displacement
        
        # padding
        in2_pad = F.pad(in2, (self.pad_size, self.pad_size, self.pad_size, self.pad_size))
        
        out_channels = (max_d * 2 + 1) ** 2
        out = torch.zeros((b, out_channels, h, w), device=in1.device, dtype=in1.dtype)
        
        idx = 0
        for i in range(-max_d, max_d + 1):
            for j in range(-max_d, max_d + 1):
                # crop in2
                in2_crop = in2_pad[:, :, 
                                   self.pad_size + i : self.pad_size + i + h, 
                                   self.pad_size + j : self.pad_size + j + w]
                
                # compute correlation
                corr = torch.sum(in1 * in2_crop, dim=1)
                out[:, idx, :, :] = corr * self.corr_multiply
                idx += 1
                
        return out
