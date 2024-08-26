"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from .pscan import pscan

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank= 'auto',
                 d_conv = 4, conv_bias=True, bias=False, scan=True):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model  = d_model
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.d_conv = d_conv
        self.d_state = d_state
        self.bias = bias
        self.conv_bias = conv_bias
        self.scan = scan

        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, hidden_state=None):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
            hidden_state: (conv_state, ssm_state)
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        if hidden_state is None:
            conv_state, ssm_state = None, None
        else:
            conv_state, ssm_state = hidden_state
            # cut state
            conv_state, ssm_state = conv_state[:b], ssm_state[:b]

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        org_x = x
        if conv_state is None:
            x = self.conv1d(x)[:, :, :l]
        else:
            x = self.conv1d(torch.cat([conv_state, x], dim=-1))[:, :, self.d_conv:l+self.d_conv]
        if self.training:
            if conv_state is None:
                conv_state = org_x.new_zeros(b, self.d_inner, self.d_conv)
            conv_state = torch.cat([conv_state[..., 1:], org_x[:, :, :1]], dim=-1)
        else:
            conv_state = org_x[:, :, -self.d_conv:]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y, ssm_state = self.ssm(x, ssm_state)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output, (conv_state.detach(), ssm_state.detach())

    def ssm(self, x, hidden=None):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y, ssm_state = self.selective_scan(x, delta, A, B, C, D, hidden)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y, ssm_state

    def selective_scan(self, u, delta, A, B, C, D, hidden=None):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        if hidden is None:
            x = torch.zeros((b, d_in, n), device=deltaA.device) # hidden state
        else:
            x = hidden

        if self.scan:
            if hidden is not None:
                deltaB_u[:, 0] += deltaA[:, 0] * hidden
            hs = pscan(deltaA, deltaB_u)  # (B, L, D, N)
            if self.training:
                ssm_state = hs[:, -1]
            else:
                ssm_state = hs[:, b-1]
            y = (hs @ C.unsqueeze(-1)).squeeze(-1)  # (B, L, D, N) @ (B, L, N, 1) -> (B, L, D, 1)
            if D is not None:
                y = y + u * D
        else:
            ys = []
            for i in range(l):
                # h_{t+1} = dA * h_t + dB * x_t
                x = deltaA[:, i] * x + deltaB_u[:, i]
                if self.training and i == 0:
                    ssm_state = x
                elif not self.training and i == l - 1:
                    ssm_state = x
                # y_{t+1} = C * h_{t+1}
                y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
                ys.append(y)
            y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
            # additional add: y_{t+1} = y_{t+1} + D * x_{t}
            if D is not None:
                y = y + u * D

        return y, ssm_state

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


