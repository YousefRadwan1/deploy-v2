"""
stage2_model.py
===============
Wav2Vec2-Base + AttentionMerge + ECAPA backbone + AAMSoftmax head.
Architecture is identical to the training notebook so checkpoint weights load correctly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


# ── Building blocks ───────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1), nn.ReLU(),
            nn.Conv1d(bottleneck, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class ECAPABlock(nn.Module):
    def __init__(self, channels, scale=8, dilation=1):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 3, dilation=dilation, padding=dilation)
            for _ in range(scale - 1)
        ])
        self.se = SEBlock(channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        out, sp = [spx[0]], None
        for i in range(self.scale - 1):
            sp = spx[i + 1] if i == 0 else sp + spx[i + 1]
            sp = self.convs[i](sp)
            out.append(sp)
        return F.relu(self.bn(self.se(torch.cat(out, dim=1)) + x))


class DifferentialAttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 8), nn.ReLU(),
            nn.Linear(channels // 8, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        mu   = x.mean(dim=1)
        attn = self.attention(mu)
        mu2  = mu * attn
        sig  = torch.std(x * attn.unsqueeze(1), dim=1)
        return torch.cat([mu2, sig], dim=-1)


class ECAPABackbone(nn.Module):
    def __init__(self, d_model=768, channels=512, embed_dim=512):
        super().__init__()
        self.conv1  = nn.Conv1d(d_model, channels, 5, padding=2)
        self.bn1    = nn.BatchNorm1d(channels)
        self.layer1 = ECAPABlock(channels, dilation=2)
        self.layer2 = ECAPABlock(channels, dilation=3)
        self.layer3 = ECAPABlock(channels, dilation=4)
        self.conv2  = nn.Conv1d(channels * 3, 1024, 1)
        self.bn2    = nn.BatchNorm1d(1024)
        self.pool   = DifferentialAttentionPool(1024)
        self.fc1    = nn.Linear(2048, embed_dim)
        self.drop   = nn.Dropout(0.4)

    def forward(self, x):
        # x: [B, T, d_model]
        x  = F.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        xa = F.relu(self.bn2(self.conv2(torch.cat([x1, x2, x3], dim=1)))).transpose(1, 2)
        return self.drop(F.relu(self.fc1(self.pool(xa))))   # [B, 512]


class AttentionMerge(nn.Module):
    def __init__(self, n_layers=12, d_model=768):
        super().__init__()
        self.squeeze = nn.Linear(d_model, 1, bias=False)
        self.gate    = nn.Linear(n_layers, n_layers)

    def forward(self, hidden_states):
        stacked  = torch.stack(hidden_states, dim=2)           # [B, T, 12, 768]
        avg_t    = stacked.mean(dim=1)                         # [B, 12, 768]
        squeezed = self.squeeze(avg_t).squeeze(-1)             # [B, 12]
        attn_w   = torch.softmax(self.gate(squeezed), dim=-1)  # [B, 12]
        merged   = (stacked * attn_w.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
        return merged                                          # [B, T, 768]


class AAMSoftmax(nn.Module):
    def __init__(self, in_features, n_classes, s=20.0, m=0.35):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def get_logits(self, x):
        return F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1)) * self.s


class W2VECAModel(nn.Module):
    def __init__(self, w2v2_model, embed_dim=512):
        super().__init__()
        self.w2v2      = w2v2_model
        self.merge     = AttentionMerge(n_layers=12, d_model=768)
        self.backbone  = ECAPABackbone(d_model=768, channels=512, embed_dim=embed_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
        )

    def _extract_features(self, x):
        with torch.no_grad():
            out = self.w2v2(x, output_hidden_states=True)
        hidden = list(out.hidden_states[1:])
        return self.merge(hidden)

    def forward(self, x):
        feats = self._extract_features(x)   # [B, T, 768]
        emb   = self.backbone(feats)         # [B, 512]
        return emb


# ── Factory ───────────────────────────────────────────────────────────────────
def build_stage2_model(device: torch.device):
    """
    Build and return (model, aam_head) with weights NOT yet loaded.
    Caller must load the checkpoint.
    """
    w2v2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    w2v2.eval()
    for p in w2v2.parameters():
        p.requires_grad = False

    model    = W2VECAModel(w2v2, embed_dim=512).to(device)
    aam_head = AAMSoftmax(512, n_classes=4, s=20.0, m=0.35).to(device)
    return model, aam_head
