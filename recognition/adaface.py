r"""
adaface.py
----------
AdaFace head — Quality-Adaptive Margin Softmax.

Reference:
    Kim et al., AdaFace: Quality Adaptive Margin for Face Recognition, CVPR 2022.
    https://arxiv.org/abs/2204.00964
    https://github.com/mk-minchul/AdaFace

In fine-tuning, we typically start with a low learning rate so AdaFace's
adaptive behavior continues from where the pretrained model left off.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceHead(nn.Module):
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 10572,
        s: float = 64.0,
        m: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 1.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.eps = 1e-3

        self.kernel = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_normal_(self.kernel)

        # Running statistics for batch norm of feature norms
        self.register_buffer("batch_mean", torch.ones(1) * 20.0)
        self.register_buffer("batch_std", torch.ones(1) * 100.0)

    def forward(self, embeddings, norms, labels):
        kernel_norm = F.normalize(self.kernel, dim=1)
        cosine = F.linear(embeddings, kernel_norm).clamp(-1 + self.eps, 1 - self.eps)

        safe_norms = torch.clip(norms, min=0.001, max=100.0).clone().detach()
        if self.training:
            with torch.no_grad():
                mean = safe_norms.mean()
                std = safe_norms.std()
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = torch.clip(margin_scaler * self.h, -1.0, 1.0)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Angular margin (inside acos)
        g_angular = self.m * margin_scaler * -1.0
        m_arc = one_hot * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # Additive margin (subtract from cosine)
        g_additive = self.m + (self.m * margin_scaler)
        m_cos = one_hot * g_additive
        cosine = cosine - m_cos

        return cosine * self.s


if __name__ == "__main__":
    head = AdaFaceHead(512, 100)
    emb = F.normalize(torch.randn(4, 512), dim=1)
    norm = torch.randn(4, 1).abs() * 20 + 20
    lbl = torch.randint(0, 100, (4,))
    print(f"Output: {tuple(head(emb, norm, lbl).shape)}")
