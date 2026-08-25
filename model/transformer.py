import copy
import math
import torch
from torch import nn
from torch.nn import Module, Dropout
from model.procrustes import SoftProcrustesLayer
import numpy as np
import random
from scipy.spatial.transform import Rotation


class PositionEncoding(nn.Module):

    def __init__(self, feature_dim, pe_type="rotary"):
        super().__init__()
        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """ combine feature and position code
        """
        if pe_type == 'rotary':
            return PositionEncoding.embed_rotary(x, pe[..., 0], pe[..., 1])
        elif pe_type == 'sinusoidal':
            return x + pe
        else:
            raise KeyError()

    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        x_position, y_position, z_position = XYZ[..., 0:1], XYZ[..., 1:2], XYZ[..., 2:3]
        div_term = torch.exp(torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device) * (
                -math.log(10000.0) / (self.feature_dim // 3)))
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal':
            position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)

        elif self.pe_type == "rotary":
            # sin/cos [θ0,θ1,θ2......θd/6-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map(
                lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                [sinx, cosx, siny, cosy, sinz, cosz])
            sin_pos = torch.cat([sinx, siny, sinz], dim=-1)
            cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)
            position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        else:
            raise KeyError()

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class GeometryAttentionLayer(nn.Module):
    def __init__(self, feature_dim, n_head, pe_type='rotary'):
        super(GeometryAttentionLayer, self).__init__()
        d_model = feature_dim
        self.nhead = n_head
        self.dim = d_model // self.nhead
        self.pe_type = pe_type
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None):

        bs = x.size(0)
        q, k, v = x, source, source
        qp, kvp = x_pe, source_pe
        q_mask, kv_mask = x_mask, source_mask

        if self.pe_type == 'sinusoidal':
            # w(x+p), attention is all you need : https://arxiv.org/abs/1706.03762
            if qp is not None:  # disentangeld
                q = q + qp
                k = k + kvp
            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'rotary':
            # Rwx roformer : https://arxiv.org/abs/2104.09864

            qw = self.q_proj(q)
            kw = self.k_proj(k)
            vw = self.v_proj(v)

            if qp is not None:  # disentangeld
                q_cos, q_sin = qp[..., 0], qp[..., 1]
                k_cos, k_sin = kvp[..., 0], kvp[..., 1]
                qw = PositionEncoding.embed_rotary(qw, q_cos, q_sin)
                kw = PositionEncoding.embed_rotary(kw, k_cos, k_sin)

            qw = qw.view(bs, -1, self.nhead, self.dim)
            kw = kw.view(bs, -1, self.nhead, self.dim)
            vw = vw.view(bs, -1, self.nhead, self.dim)

        else:
            raise KeyError()

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        if kv_mask is not None:
            a.masked_fill_(q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))
        a = a / qw.size(3) ** 0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e


class GeometryMaskAttentionLayer(nn.Module):
    def __init__(self, feature_dim, n_head, pe_type='rotary'):
        super(GeometryMaskAttentionLayer, self).__init__()
        d_model = feature_dim
        self.nhead = n_head
        self.dim = d_model // self.nhead
        self.pe_type = pe_type
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.nhead * 3 , self.nhead * 3, bias=False),
            nn.LeakyReLU(True),
            nn.Linear(self.nhead* 3, 3, bias=False),
        )

        # norm and dropout
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

    def forward(self, x, tgt, x_pe, tgt_pe, mask1, mask2, xyz2):
        bs = x.size(0)
        q, k, v = x, tgt, xyz2.unsqueeze(-1).repeat([1, 1, 1, self.nhead])
        qp, kvp = x_pe, tgt_pe
        q = self.norm_q(q)
        k = self.norm_k(k)
        # Rwx roformer : https://arxiv.org/abs/2104.09864
        qw = self.q_proj(q)
        kw = self.k_proj(k)

        if qp is not None:  # disentangeld
            q_cos, q_sin = qp[..., 0], qp[..., 1]
            k_cos, k_sin = kvp[..., 0], kvp[..., 1]
            qw = PositionEncoding.embed_rotary(qw, q_cos, q_sin)
            kw = PositionEncoding.embed_rotary(kw, k_cos, k_sin)

        qw = qw.view(bs, -1, self.nhead, self.dim)
        kw = kw.view(bs, -1, self.nhead, self.dim)

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        a = a / qw.size(3) ** 0.5
        a = a * mask1.unsqueeze(-1) * mask2.unsqueeze(1)
        a = torch.softmax(a, dim=2)
        output = torch.einsum("nqwh,nwsh->nqsh", a, v).contiguous()  # [B, N, (3, D)]
        output = output.view(bs, -1, self.nhead * 3)  # [N, L, C]
        output = self.mlp(output)
        return output


class Transformer_matching_mlp(nn.Module):
    def __init__(self, feature_dim, n_head, pe_type='rotary'):
        super(Transformer_matching_mlp, self).__init__()
        self.d_model = feature_dim
        self.nhead = n_head
        self.pe_type = pe_type
        self.positional_encoding = PositionEncoding(feature_dim=self.d_model, pe_type="rotary")
        self.selfAttentionSrc = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.selfAttentionTgt = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.crossAttention = GeometryAttentionLayer(feature_dim=self.d_model, n_head=self.nhead, pe_type='rotary')
        self.corr_mlp = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model, out_features=3, bias=False),
        )

    def forward(self, src_feat, tgt_feat, s_pcd, t_pcd):
        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"
        src_pe = self.positional_encoding(s_pcd)
        tgt_pe = self.positional_encoding(t_pcd)
        src_feat = self.selfAttentionSrc(src_feat, src_feat, src_pe, src_pe)
        tgt_feat = self.selfAttentionTgt(tgt_feat, tgt_feat, tgt_pe, tgt_pe)
        src_feat_cross = self.crossAttention(src_feat, tgt_feat, src_pe, tgt_pe)
        position = self.corr_mlp(src_feat_cross)
        return position


class RepositioningTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.d_model = config['feature_dim']
        self.nhead = config['n_head']
        self.layer_types = config['layer_types']
        self.positioning_type = config['positioning_type']
        self.pe_type = config['pe_type']

        self.entangled = config['entangled']

        self.positional_encoding = PositionEncoding(config)

        encoder_layer = GeometryAttentionLayer(config)

        self.layers = nn.ModuleList()

        for l_type in self.layer_types:

            if l_type in ['self', 'cross']:

                self.layers.append(copy.deepcopy(encoder_layer))

            elif l_type == "positioning":

                if self.positioning_type == 'procrustes':
                    positioning_layer = nn.ModuleList()
                    # positioning_layer.append( Matching(config['feature_matching']))
                    positioning_layer.append(SoftProcrustesLayer(config['procrustes']))
                    self.layers.append(positioning_layer)

                elif self.positioning_type in ['oracle', 'randSO3']:
                    self.layers.append(None)

                else:
                    raise KeyError(self.positioning_type + " undefined positional encoding type")

            else:
                raise KeyError()

        self._reset_parameters()

    def forward(self, src_feat, tgt_feat, s_pcd, t_pcd, src_mask, tgt_mask, data, T=None, timers=None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        if T is not None:
            R, t = T
            src_pcd_wrapped = (torch.matmul(R, s_pcd.transpose(1, 2)) + t).transpose(1, 2)
            tgt_pcd_wrapped = t_pcd
        else:
            src_pcd_wrapped = s_pcd
            tgt_pcd_wrapped = t_pcd

        src_pe = self.positional_encoding(src_pcd_wrapped)
        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)

        if not self.entangled:

            position_layer = 0
            data.update({"position_layers": {}})

            for layer, name in zip(self.layers, self.layer_types):

                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, src_pe, src_pe, src_mask, src_mask, )
                    tgt_feat = layer(tgt_feat, tgt_feat, tgt_pe, tgt_pe, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')

                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, src_pe, tgt_pe, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, tgt_pe, src_pe, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')

                elif name == 'positioning':

                    if self.positioning_type == 'procrustes':

                        conf_matrix, match_pred = layer[0](src_feat, tgt_feat, src_pe, tgt_pe, src_mask, tgt_mask, data,
                                                           pe_type=self.pe_type)

                        position_layer += 1
                        data["position_layers"][position_layer] = {"conf_matrix": conf_matrix, "match_pred": match_pred}

                        if self.timers: self.timers.tic('procrustes_layer')
                        R, t, R_forwd, t_forwd, condition, solution_mask = layer[1](conf_matrix, s_pcd, t_pcd, src_mask,
                                                                                    tgt_mask)
                        if self.timers: self.timers.toc('procrustes_layer')

                        data["position_layers"][position_layer].update({
                            "R_s2t_pred": R, "t_s2t_pred": t, "solution_mask": solution_mask, "condition": condition})

                        src_pcd_wrapped = (torch.matmul(R_forwd, s_pcd.transpose(1, 2)) + t_forwd).transpose(1, 2)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    elif self.positioning_type == 'randSO3':
                        src_pcd_wrapped = self.rand_rot_pcd(s_pcd, src_mask)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    elif self.positioning_type == 'oracle':
                        # Note R,t ground truth is only available for computing oracle position encoding
                        rot_gt = data['batched_rot']
                        trn_gt = data['batched_trn']
                        src_pcd_wrapped = (torch.matmul(rot_gt, s_pcd.transpose(1, 2)) + trn_gt).transpose(1, 2)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    else:
                        raise KeyError(self.positioning_type + " undefined positional encoding type")

                else:
                    raise KeyError

            return src_feat, tgt_feat, src_pe, tgt_pe

        else:  # pos. fea. entangeled

            position_layer = 0
            data.update({"position_layers": {}})

            src_feat = PositionEncoding.embed_pos(self.pe_type, src_feat, src_pe)
            tgt_feat = PositionEncoding.embed_pos(self.pe_type, tgt_feat, tgt_pe)

            for layer, name in zip(self.layers, self.layer_types):
                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, None, None, src_mask, src_mask, )
                    tgt_feat = layer(tgt_feat, tgt_feat, None, None, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')
                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, None, None, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, None, None, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')
                elif name == 'positioning':
                    pass

            return src_feat, tgt_feat, src_pe, tgt_pe

    def rand_rot_pcd(self, pcd, mask):
        '''
        @param pcd: B, N, 3
        @param mask: B, N
        @return:
        '''

        pcd[~mask] = 0.
        N = mask.shape[1]
        n_points = mask.sum(dim=1, keepdim=True).view(-1, 1, 1)
        bs = pcd.shape[0]

        euler_ab = np.random.rand(bs, 3) * np.pi * 2  # anglez, angley, anglex
        rand_rot = torch.from_numpy(Rotation.from_euler('zyx', euler_ab).as_matrix()).to(pcd)
        pcd_u = pcd.mean(dim=1, keepdim=True) * N / n_points
        pcd_centered = pcd - pcd_u
        pcd_rand_rot = torch.matmul(rand_rot, pcd_centered.transpose(1, 2)).transpose(1, 2) + pcd_u
        return pcd_rand_rot

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
