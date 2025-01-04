import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from model_utils.mamba import Mamba, RMSNorm
from model_utils.hp_filter import hp_filter


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class MovingAvgFilter(nn.Module):
    @staticmethod
    def period_estimate(x, top_k=1):
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x.permute(0, 2, 1).contiguous(), dim=-1)
        x_psd = x_fft * torch.conj(x_fft)  # PSD
        est_freq = torch.unique(torch.topk(x_psd.abs().mean([0, 1])[1:L//2], top_k)[1] + 1)
        est_period = [int(L / f) for f in est_freq]

        return est_period

    def __init__(self, top_k=1):
        super(MovingAvgFilter, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        B, L, D = x.shape
        est_periods = self.period_estimate(x, self.top_k)
        trend_hats = []
        for p in est_periods:
            moving_avg = torch.nn.functional.avg_pool1d(x.transpose(-1, -2), kernel_size=p, stride=1, padding=0).transpose(-1, -2)
            front = (p - 1) // 2 + int(p % 2 == 0)
            back = (p - 1) // 2
            trend_hat = torch.cat([
                moving_avg[:, 0].unsqueeze(1).expand(-1, front, -1),
                moving_avg,
                moving_avg[:, -1].unsqueeze(1).expand(-1, back, -1),
            ], dim=1)
            trend_hats.append(trend_hat)
        trend_hats = torch.stack(trend_hats).mean(0)
        res = x - trend_hats

        return res, trend_hats

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, conv_bias=True, bias=False, scan=True):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv
        self.d_state = d_state
        self.bias = bias
        self.conv_bias = conv_bias
        self.scan = scan

        self.norm = RMSNorm(d_model)
        self.s6 = Mamba(d_model, d_state, expand, dt_rank, d_conv, conv_bias, bias, scan)

    def forward(self, x, hidden=None):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        x = self.norm(x)
        x, hidden = self.s6(x, hidden)
        return x, hidden

class Hpfilter(object):
    def __init__(self, lam=1e8, warm_up_step=50):
        self.lam = lam
        self.warm_up_step = warm_up_step

    def __call__(self, x, hidden=None):
        bsz, tsz, dsz = x.shape
        batch_x = x.transpose(-1, -2).reshape(-1, tsz)

        # if hidden is None:
        trend, hidden = hp_filter(batch_x, lam=self.lam, warm_up_step=self.warm_up_step)
        # else:
        #     trend, hidden = hp_filter(batch_x, lam=self.lam, warm_up_step=0, hidden=hidden)

        trend = trend.reshape(bsz, dsz, tsz).transpose(-1, -2)

        return trend, hidden


class DecomposeMambaSSM(nn.Module):
    def __init__(self, input_size, window_size, d_model=32, state_size=16,
                 expand=2, block_num=3, pre_filter=True, decomp=True
                 ):
        super(DecomposeMambaSSM, self).__init__()
        self.window_size = window_size
        self.block_num = block_num
        self.pre_filter = pre_filter
        if self.pre_filter:
            self.decompose = Hpfilter(lam=1e4, warm_up_step=window_size)
        self.embedding = TokenEmbedding(input_size, d_model)

        self.mix_blocks = nn.ModuleList(
            MambaBlock(d_model, state_size, expand) for _ in range(block_num)
        )
        self.seasonal_projection = nn.Conv1d(in_channels=d_model, out_channels=input_size, kernel_size=3, stride=1,
                                             padding=1, padding_mode='replicate', bias=False)
        self.trend_projection = nn.Conv1d(in_channels=d_model, out_channels=input_size, kernel_size=3, stride=1,
                                          padding=1, padding_mode='replicate', bias=False)

        self.decomp = decomp
        if decomp:
            self.moving_avg = MovingAvgFilter(top_k=1)

    def forward(self, x, hidden=None, **kwargs):
        if hidden is not None:
            ssm_hidden, trend_hidden = hidden
        else:
            ssm_hidden, trend_hidden = None, None
        if self.pre_filter:
            if self.training or trend_hidden is None:
                trend_init, trend_hidden = self.decompose(x)
                seasonal_init = x - trend_init
            else:
                trend_init, trend_hidden = self.decompose(x, hidden=trend_hidden)
                seasonal_init = x - trend_init
        else:
            seasonal_init = x
            trend_init = 0
        x = self.embedding(seasonal_init)

        new_hidden = []
        trends = 0
        if ssm_hidden is None:
            ssm_hidden = [None] * self.block_num
        for block, h in zip(self.mix_blocks, ssm_hidden):
            x, new_h = block(x, h)
            if self.decomp:
                x, block_trend = self.moving_avg(x)
                trends += block_trend

            new_hidden.append(new_h)

        if self.decomp:
            trends = self.trend_projection(trends.permute(0, 2, 1)).transpose(1, 2)
        seasonal = self.seasonal_projection(x.permute(0, 2, 1)).transpose(1, 2)

        if not self.pre_filter and not self.decomp:
            trends = torch.zeros_like(seasonal)
        else:
            trends += trend_init

        rec = trends + seasonal

        res = {
            "x": rec,
            "hidden": (new_hidden, trend_hidden),
        }

        return res

    @property
    def metric_tags(self):
        tags = ["total_loss", "reconstruction_loss"]

        return tags

    def cal_loss(self, res, target, epoch, **kwargs):
        x, _ = res.values()
        rec_loss = nn.functional.mse_loss(target, x)
        loss = rec_loss
        metrics = (loss.item(),)
        count = (1,)
        metrics += (rec_loss.item(),)
        count += (1,)

        return loss, metrics, count

    def anomaly_detection(self, test_dataloader, device="cpu"):
        self.eval()
        scores = []
        y_trues = []
        y_hats = []
        hidden = None
        with torch.no_grad():
            for data, target, data_lens, _ in tqdm(test_dataloader):
                if device != "cpu":
                    data = data.cuda(0)
                    target = target.cuda(0)
                x, hidden = self.forward(data, data_lens, hidden).values()
                score = nn.functional.mse_loss(target, x, reduction="none")
                # b, t -> b
                y_trues.append(data.cpu().detach().numpy().reshape(-1, x.shape[-1]))
                y_hats.append(x.cpu().detach().numpy().reshape(-1, x.shape[-1]))
                scores.append(score.cpu().detach().numpy().reshape(-1, score.shape[-1]))
        # b, t -> b
        y_trues = np.concatenate(y_trues, axis=0)
        y_hats = np.concatenate(y_hats, axis=0)
        scores = np.concatenate(scores, axis=0).mean(axis=-1)

        label = test_dataloader.dataset.label
        scores = scores[:label.shape[0]]
        y_trues = y_trues[:label.shape[0]]
        y_hats = y_hats[:label.shape[0]]

        return scores, label, y_trues, y_hats