# Last modified: 2026-02-03 (Fixed Pandas read-only error)

import pandas as pd
import torch
import torch.nn.functional as F
import torch.fft
import inspect

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        # 1. 显式指定 dtype=float，防止整数类型推断
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"], dtype=float)
        self.reset()

    def reset(self):
        # 2. 【核心修复】使用 iloc 全表赋值，解决 read-only 报错
        self._data.iloc[:, :] = 0.0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        # 3. 使用 loc 进行安全更新
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]

    def avg(self, key):
        return self._data.loc[key, "average"]

    def result(self):
        return self._data["average"].to_dict()

def _input_check(batch, ref1=None, ref2=None):  # Check input
    def _batch_check(batch):    
        assert type(batch) is torch.Tensor, "input is not a tensor"

        if len(batch.shape) == 2:
            batch = batch.unsqueeze(0)
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)

        assert len(batch.shape) == 4, "dimension number error"  
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
        
        assert batch.shape[1] == 3, "channel number error"
        # Only guard genuinely empty images. SmoothFusion-style evaluation
        # tensors are in [0, 1], so ``max < 1`` would inject noise into normal
        # dark images.
        if torch.max(batch) <= 1e-12:
            noise = torch.randn_like(batch) * 0.1 + 1  # Add random noise with mean 1
            batch = batch + noise
        return batch
    
    batch = _batch_check(batch).to(torch.float64)
    if ref1 is not None:
        ref1 = _batch_check(ref1).to(torch.float64).to(batch.device)
    if ref2 is not None:
        ref2 = _batch_check(ref2).to(torch.float64).to(batch.device)

    return batch, ref1, ref2


def Metric_VIF(batch, ref1=None, ref2=None):
    def _vifp_batch(ref, dist):
        ref = ref.to(torch.float64)
        dist = dist.to(torch.float64)
        sigma_nsq = 2.0  # Visual noise variance
        eps = 1e-10
        num = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)
        den = torch.zeros(ref.shape[0], ref.shape[1]).to(dist.device)

        for scale in range(1, 5):  # 4 scales
            N = 2**(4 - scale + 1) + 1
            sd = N / 5.0
            win = _fspecial_gaussian_torch(N, sd).to(dist.device)

            if scale > 1:
                ref = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])[:, :, ::2, ::2]
                dist = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])[:, :, ::2, ::2]

            mu1 = F.conv2d(ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu2 = F.conv2d(dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1])
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(ref * ref, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu1_sq
            sigma2_sq = F.conv2d(dist * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu2_sq
            sigma12 = F.conv2d(ref * dist, win.repeat(batch.shape[1], 1, 1, 1), padding=0, groups=batch.shape[1]) - mu1_mu2
            sigma1_sq = torch.clamp(sigma1_sq, min=0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0)
            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)), dim=(2,3))
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=(2,3))

        vifp_val = num / den
        vifp_val = _ignore_nan(vifp_val)
        return vifp_val
    
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        vifp1 = torch.mean(_vifp_batch(ref1, batch), dim=1)  # (B,)
        vifp2 = torch.mean(_vifp_batch(ref2, batch), dim=1)  # (B,)
        return (vifp1 + vifp2) / 2.0 

def _filter2d_gaussian(img, win):
    pad = (win.shape[-2] // 2, win.shape[-1] // 2, win.shape[-2] // 2, win.shape[-1] // 2)
    img = F.pad(img, pad=pad, mode='reflect')
    return F.conv2d(img, win.repeat(img.shape[1], 1, 1, 1), padding=0, groups=img.shape[1])


def _ssim_pair(img, img_ref):
    """Per-sample SSIM on the 0-255 scale. Input [N,C,H,W] -> [N]."""
    K1 = (0.01 * 255) ** 2
    K2 = (0.03 * 255) ** 2
    win = _fspecial_gaussian_torch(11, 1.5).to(img.device)

    mu1 = _filter2d_gaussian(img, win)
    mu2 = _filter2d_gaussian(img_ref, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filter2d_gaussian(img * img, win) - mu1_sq
    sigma2_sq = _filter2d_gaussian(img_ref * img_ref, win) - mu2_sq
    sigma12 = _filter2d_gaussian(img * img_ref, win) - mu1_mu2

    ssim_map = ((2. * mu1_mu2 + K1) * (2. * sigma12 + K2)) / \
        ((mu1_sq + mu2_sq + K1) * (sigma1_sq + sigma2_sq + K2))
    return torch.mean(ssim_map, dim=(2, 3))


def Metric_SSIM(batch, ref1=None, ref2=None):
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        output = (_ssim_pair(batch, ref1) + _ssim_pair(batch, ref2)) / 2
        return torch.mean(output, dim=1)


def _smoothfusion_inputs(batch, ref1, ref2):
    """Prepare metric inputs using SmoothFusion's [0, 1] convention."""
    tensors = []
    for image in (batch, ref1, ref2):
        if image is None:
            raise ValueError("SmoothFusion metrics require two reference images")
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"metric inputs expect [B,C,H,W], got {tuple(image.shape)}")
        image = image.to(dtype=torch.float64)
        if image.numel() and image.detach().amax() > 1.0:
            image = image / 255.0
        tensors.append(image)

    channels = max(image.shape[1] for image in tensors)
    if channels not in (1, 3):
        raise ValueError(f"metric inputs must have 1 or 3 channels, got {channels}")
    if channels == 3:
        tensors = [image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
                   for image in tensors]
    return tuple(tensors)


def Metric_MSE(batch, ref1, ref2):
    """SmoothFusion MSE: average source fidelity error on the [0, 1] scale."""
    with torch.no_grad():
        batch, ref1, ref2 = _smoothfusion_inputs(batch, ref1, ref2)
        mse1 = (batch - ref1).pow(2).mean(dim=(1, 2, 3))
        mse2 = (batch - ref2).pow(2).mean(dim=(1, 2, 3))
        return 0.5 * (mse1 + mse2)


def _smoothfusion_ncc_pair(target, prediction):
    target = target.reshape(target.shape[0], -1)
    prediction = prediction.reshape(prediction.shape[0], -1)
    target_mean = target.mean(dim=1, keepdim=True)
    prediction_mean = prediction.mean(dim=1, keepdim=True)
    target_var = ((target - target_mean) ** 2).mean(dim=1)
    prediction_var = ((prediction - prediction_mean) ** 2).mean(dim=1)
    covariance = ((target - target_mean) * (prediction - prediction_mean)).mean(dim=1)
    return covariance / torch.sqrt((target_var + 1e-5) * (prediction_var + 1e-6))


def Metric_NCC(batch, ref1, ref2):
    """SmoothFusion NCC: average Pearson correlation with both sources."""
    with torch.no_grad():
        batch, ref1, ref2 = _smoothfusion_inputs(batch, ref1, ref2)
        ncc1 = _smoothfusion_ncc_pair(ref1, batch)
        ncc2 = _smoothfusion_ncc_pair(ref2, batch)
        return 0.5 * (ncc1 + ncc2)


def Metric_Redge(batch, ref1, ref2):
    """Relative edge retention used by the HDO registration report.

    The metric compares the soft Sobel edge strength retained in the result
    against each source and averages the two retention ratios.  It is exposed
    under the ``R_edge`` report name; the paper's private evaluator should be
    used when exact Table 3 reproduction is required.
    """
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float64, device=batch.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)

        def edge_strength(image):
            gray = rgb2gray(image)
            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            return torch.sqrt(gx.square() + gy.square() + 1e-12)

        result_edge = edge_strength(batch)
        source_edges = (edge_strength(ref1), edge_strength(ref2))
        ratios = []
        for source_edge in source_edges:
            retained = torch.minimum(result_edge, source_edge).sum(dim=(1, 2, 3))
            source_total = source_edge.sum(dim=(1, 2, 3)).clamp_min(1e-12)
            ratios.append(retained / source_total)
        return 0.5 * (ratios[0] + ratios[1])


# ----------------------------------------------------------------------
# 时间一致性指标
#
# 说明：SmoothFusion/HDO 表 3 报告 ITF 与 T-SSIM，但当前公开仓库
# （E:\res and fus\SmoothFusion）没有提供它们的原始评测函数。因此下面两个按论文
# 指标语义实现，用于统一训练/评测口径；若要逐位复现论文数值，仍需要作者原始脚本。
#
# 另需注意该论文自己的警告：VideoFusion/ReCoNet 在时间指标上数值更好，但那是
# 因为严重错位与大范围伪影人为压低了帧间差，属于"假的时间平滑"。所以时间指标
# 必须与空间保真度指标（VIF/SSIM/MI/Qabf）一起报告，不能单独看。
# ----------------------------------------------------------------------

def _gray_sequence(seq):
    """[B,T,C,H,W] -> [B,T,1,H,W] float64（0-255 尺度）。"""
    if seq.dim() != 5:
        raise ValueError(f"temporal metrics expect [B,T,C,H,W], got {tuple(seq.shape)}")
    batch, time, channels, height, width = seq.shape
    flat = seq.reshape(batch * time, channels, height, width)
    gray = rgb2gray(flat)
    return gray.reshape(batch, time, 1, height, width)


def Metric_ITF(batch, ref1=None, ref2=None):
    """ITF: 融合视频相邻帧的平均绝对差（0-255 灰度尺度）。越低越平滑。

    这是对 SmoothFusion "inter-frame time fluctuation" 的本方定义：直接度量
    输出视频的帧间波动能量。注意它可被模糊/错位人为压低，需与空间指标同看。
    """
    with torch.no_grad():
        seq = _gray_sequence(batch.to(torch.float64))
        if seq.shape[1] < 2:
            return torch.zeros(seq.shape[0], device=seq.device, dtype=torch.float64)
        return (seq[:, 1:] - seq[:, :-1]).abs().mean(dim=(1, 2, 3, 4))


def Metric_TSSIM(batch, ref1=None, ref2=None):
    """T-SSIM: 融合视频相邻帧之间的 SSIM 均值。越高越稳定。

    这是对 SmoothFusion "temporal structure similarity" 的本方定义。
    输入需要 [B,T,C,H,W] 且 T >= 2（预测序列本身的时间维，走整段而不是中心帧）。
    """
    with torch.no_grad():
        seq = batch.to(torch.float64)
        if seq.dim() != 5:
            raise ValueError(f"Metric_TSSIM expects [B,T,C,H,W], got {tuple(seq.shape)}")
        batch_size, time, channels, height, width = seq.shape
        if time < 2:
            return torch.zeros(batch_size, device=seq.device, dtype=torch.float64)
        left = seq[:, :-1].reshape(batch_size * (time - 1), channels, height, width)
        right = seq[:, 1:].reshape(batch_size * (time - 1), channels, height, width)
        values = _ssim_pair(left, right)
        return values.reshape(batch_size, time - 1).mean(dim=1)


def Metric_NMI(batch, ref1, ref2):
    """NMI: 融合图与两个源图的归一化互信息均值。~1.0 附近为正常量级。"""
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        nmi1, _ = _torch_normalized_mutual_info_score(batch, ref1)
        nmi2, _ = _torch_normalized_mutual_info_score(batch, ref2)
    return 0.5 * (torch.mean(nmi1, dim=1) + torch.mean(nmi2, dim=1))


def Metric_LNCC(batch, ref1, ref2, window=17, eps=1e-5):
    """LNCC: 局部归一化互相关。

    定义逐行对齐 SmoothFusion 官方代码 ``AKRF/utils/util.py`` 的 ``LNCC`` 类：
    用 ones 滤波器在 window×window 窗口内求局部和，取**平方**相关系数
    ``cc = cross^2 / (I_var * J_var + eps)``，再对全图取均值。
    窗口默认 17、eps=1e-5，与官方一致。官方实现在单通道上计算，这里同样先转灰度。
    """
    def _lncc(source, target):
        source = rgb2gray(source)
        target = rgb2gray(target)
        filt = torch.ones(1, 1, window, window, dtype=source.dtype,
                          device=source.device)
        pad = window // 2
        win_size = window * window

        i_sum = F.conv2d(source, filt, padding=pad)
        j_sum = F.conv2d(target, filt, padding=pad)
        i2_sum = F.conv2d(source * source, filt, padding=pad)
        j2_sum = F.conv2d(target * target, filt, padding=pad)
        ij_sum = F.conv2d(source * target, filt, padding=pad)

        u_i = i_sum / win_size
        u_j = j_sum / win_size
        cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
        i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * win_size
        j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * win_size
        return (cross * cross / (i_var * j_var + eps)).mean()

    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        return 0.5 * (_lncc(ref1, batch) + _lncc(ref2, batch))
    
def Metric_MI(batch, ref1=None, ref2=None):
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        _, mi1 = _torch_normalized_mutual_info_score(batch, ref1) 
        _, mi2 = _torch_normalized_mutual_info_score(batch, ref2) 
    
    return torch.mean(mi1, dim=1) + torch.mean(mi2, dim=1)


def Metric_Qabf(batch, ref1, ref2):
    def Qabf_getArray(img):
        h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float64, device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64, device=img.device).view(1, 1, 3, 3).repeat(img.shape[1], 1, 1, 1)
        
        SAx = F.conv2d(img, h3, padding=1, groups=img.shape[1])
        SAy = F.conv2d(img, h1, padding=1, groups=img.shape[1])
        
        gA = torch.sqrt(SAx.pow(2) + SAy.pow(2))
        aA = torch.zeros_like(img)
        aA[SAx == 0] = torch.pi / 2 
        mask = SAx != 0 
        aA[mask] = torch.atan(SAy[mask] / SAx[mask])
        return gA, aA

    def Qabf_getQabf(aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8
        
        GAF = torch.where(gA > gF, gF / gA,
                        torch.where(gA < gF, gA / gF,
                                    gF))

        AAF = 1 - (torch.abs(aA - aF) / (torch.pi / 2))
        
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        
        return QAF

    def Qabf(img, img1, img2):
        gA, aA = Qabf_getArray(img1)
        gB, aB = Qabf_getArray(img2)
        gF, aF = Qabf_getArray(img)
        
        QAF = Qabf_getQabf(aA, gA, aF, gF)
        QBF = Qabf_getQabf(aB, gB, aF, gF)
        
        nume = torch.sum(QAF * gA + QBF * gB, dim=(2, 3))
        deno = torch.sum(gA + gB, dim=(2, 3))
        
        return nume / deno
    
    with torch.no_grad():
        batch, ref1, ref2 = _input_check(batch, ref1, ref2)
        results = Qabf(batch, ref1, ref2) 
        return torch.mean(results, dim=1) 
    
def _torch_normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'):
    batch_size = labels_true.shape[0]
    nmi_values = torch.zeros(batch_size, dtype=torch.float64)
    mi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    nmi_calculate = torch.zeros(labels_true.shape[0], labels_true.shape[1])
    for b in range(batch_size):
        for c in range(labels_true.shape[1]):
            lbl_true = labels_true[b, c].flatten()
            lbl_pred = labels_pred[b, c].flatten()
            
            unique_true = torch.unique(lbl_true)
            unique_pred = torch.unique(lbl_pred)
            if (len(unique_true) == len(unique_pred) == 1) or (len(unique_true) == len(unique_pred) == 0):
                nmi_values[b] = 1.0
                continue
            
            contingency = _torch_contingency_matrix(lbl_true, lbl_pred)
            contingency = contingency.to(torch.float64)
            total = contingency.sum()
            if total == 0:
                nmi_values[b] = 1.0
                continue
            
            pi = contingency.sum(dim=1)
            pj = contingency.sum(dim=0)
            log_contingency = (contingency / total).log()
            log_pi = (pi / total).log().unsqueeze(1)
            log_pj = (pj / total).log().unsqueeze(0)
            
            mi = (contingency / total) * (log_contingency - log_pi - log_pj)
            mi = mi.nansum()  
            mi_calculate[b,c] = mi
            
            if mi <= 1e-15:
                nmi_values[b] = 0.0
                continue
            
            h_true = (-(pi[pi > 0] / pi.sum()).log() * (pi[pi > 0] / pi.sum())).sum()
            h_pred = (-(pj[pj > 0] / pj.sum()).log() * (pj[pj > 0] / pj.sum())).sum()
            
            if average_method == 'arithmetic':
                normalizer = 0.5 * (h_true + h_pred)
            elif average_method == 'geometric':
                normalizer = torch.sqrt(h_true * h_pred)
            else:
                raise ValueError(f"Unsupported average_method: {average_method}")
            
            nmi_values[b] = mi / normalizer
            nmi_calculate[b,c] = nmi_values[b]
    return nmi_calculate, mi_calculate
    
def _torch_contingency_matrix(labels_true, labels_pred):
    unique_true, map_true = torch.unique(labels_true, return_inverse=True)
    unique_pred, map_pred = torch.unique(labels_pred, return_inverse=True)
    
    contingency = torch.zeros(
        (len(unique_true), len(unique_pred)),  
        dtype=torch.int64,  
        device=labels_true.device
    )
    contingency.index_put_(
        (map_true, map_pred),  
        torch.ones_like(map_true, dtype=torch.int64),  
        accumulate=True
    )
    return contingency

def _normalize(tensor):
    min_val = torch.min(tensor.flatten(start_dim=2), dim=2)[0]
    max_val = torch.max(tensor.flatten(start_dim=2), dim=2)[0]
    normalized = torch.zeros_like(tensor)
    mask = (max_val - min_val) != 0
    min_val = min_val.unsqueeze(-1).unsqueeze(-1)
    max_val = max_val.unsqueeze(-1).unsqueeze(-1)
    normalized[mask] = 255.0 * (tensor[mask] - min_val[mask]) / (max_val[mask] - min_val[mask])
    return normalized

def _fspecial_gaussian_torch(win_size, sigma):
    coords = torch.arange(-win_size // 2 + 1, win_size // 2 + 1, dtype=torch.float64)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g /= g.sum()
    kernel = g.reshape(1, 1, win_size, win_size).contiguous()
    return kernel

def rgb2gray(tensor):
    if tensor.shape[1] == 1:
        return tensor
    elif tensor.shape[1] == 3:
        r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.to(torch.float64)
    else:
        raise ValueError(f"channel must be 1 or 3, got {tensor.shape[1]}")
    
SPATIAL_METRIC_NAMES = {"Metric_MSE", "Metric_NCC", "Metric_Redge", "Metric_LNCC",
                        "Metric_VIF", "Metric_SSIM", "Metric_MI", "Metric_Qabf",
                        "Metric_NMI"}
TEMPORAL_METRIC_NAMES = {"Metric_ITF", "Metric_TSSIM"}


def compute_metrics(metric_funcs, fusion_pred, I1, I2):
    """计算指标。

    空间指标作用在预测序列的**中心帧**上（对齐到输入 5 帧窗口的中心）；
    时间指标作用在**整段**预测序列上，因此要求 fusion_pred 的 T >= 2。
    时间指标必须与空间指标一起报告：错位与模糊会人为压低 ITF、抬高 T-SSIM，
    单看时间指标会得到"假的时间平滑"。
    """
    results = {}
    for met_func in metric_funcs:
        _metric_name = met_func.__name__

        if _metric_name in TEMPORAL_METRIC_NAMES:
            _metric = met_func(torch.round(fusion_pred * 255)).item()

        elif _metric_name in SPATIAL_METRIC_NAMES:
            # MSE/NCC/LNCC/R_edge use the unquantized center frame. Legacy
            # metrics retain the historical 0-255 evaluation path.
            if _metric_name in {"Metric_MSE", "Metric_NCC", "Metric_Redge", "Metric_LNCC"}:
                fusion_clip = fusion_pred[:, 1, :, :, :]
                ir_clip = I1[:, 2, :, :, :]
                rgb_clip = I2[:, 2, :, :, :]
            else:
                fusion_clip = torch.round(fusion_pred[:, 1, :, :, :] * 255)
                ir_clip = torch.round(I1[:, 2, :, :, :] * 255)
                rgb_clip = torch.round(I2[:, 2, :, :, :] * 255)

            required = len([p for p in inspect.signature(met_func).parameters.values()
                            if p.default is inspect.Parameter.empty])
            if required == 1:
                _metric = met_func(fusion_clip).item()
            elif required == 3:
                _metric = met_func(fusion_clip, ir_clip, rgb_clip).item()
            else:
                raise ValueError(
                    f"Metric function {_metric_name} has unsupported number of "
                    f"required parameters: {required}")
        else:
            raise ValueError(f"Unsupported metric function: {_metric_name}")

        results[_metric_name] = _metric
    return results

def _ignore_nan(tensor):
    not_nan_channel = ~torch.isnan(tensor)  # B,C
    tensor_without_nan = torch.nan_to_num(tensor, nan=0.0)
    return torch.sum(tensor_without_nan, dim=1).unsqueeze(1) / torch.sum(not_nan_channel, dim=1)
