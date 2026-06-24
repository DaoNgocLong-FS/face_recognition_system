r"""
iresnet.py  (CORRECTED to match official AdaFace architecture)
--------------------------------------------------------------
IResNet backbone matching the OFFICIAL AdaFace net.py exactly, so that
pretrained weights (adaface_ir50_ms1mv2.ckpt) load correctly.

Source: https://github.com/mk-minchul/AdaFace/blob/master/net.py

CRITICAL: For IR-50 (num_layers <= 100), official code uses BasicBlockIR
(NOT a bottleneck). Standard adaface_ir50_ms1mv2.ckpt uses mode 'ir' (no SE).

forward(x) returns (embedding_l2_normalized, embedding_norm).
"""

from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear,
                      MaxPool2d, Module, PReLU, ReLU, Sequential, Sigmoid)


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlockIR(Module):
    """Used for IR-18/34/50/100 (num_layers <= 100)."""
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        return [get_block(64, 64, 2), get_block(64, 128, 2),
                get_block(128, 256, 2), get_block(256, 512, 2)]
    if num_layers == 34:
        return [get_block(64, 64, 3), get_block(64, 128, 4),
                get_block(128, 256, 6), get_block(256, 512, 3)]
    if num_layers == 50:
        return [get_block(64, 64, 3), get_block(64, 128, 4),
                get_block(128, 256, 14), get_block(256, 512, 3)]
    if num_layers == 100:
        return [get_block(64, 64, 3), get_block(64, 128, 13),
                get_block(128, 256, 30), get_block(256, 512, 3)]
    raise ValueError(f"Unsupported num_layers: {num_layers}")


class Backbone(Module):
    def __init__(self, input_size=112, num_layers=50, mode="ir", output_dim=512):
        super().__init__()
        assert input_size in (112, 224)
        assert num_layers in (18, 34, 50, 100)
        assert mode in ("ir", "ir_se")

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(num_layers)
        unit_module = BasicBlockIRSE if mode == "ir_se" else BasicBlockIR
        output_channel = 512

        if input_size == 112:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 7 * 7, output_dim),
                BatchNorm1d(output_dim, affine=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(output_channel), Dropout(0.4), Flatten(),
                Linear(output_channel * 14 * 14, output_dim),
                BatchNorm1d(output_dim, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for module in self.body:
            x = module(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True).clamp(min=1e-9)
        normalized = x / norm
        return normalized, norm


def build_iresnet(name="ir_50", embedding_size=512):
    """
    Names:
        ir_50    -> Backbone(50, 'ir')   [matches adaface_ir50_ms1mv2.ckpt]
        ir_50_se -> Backbone(50, 'ir_se')
        ir_18, ir_34, ir_100 + _se variants also supported.
    """
    name = name.lower()
    if name.endswith("_se"):
        mode = "ir_se"
        num_layers = int(name.replace("ir_", "").replace("_se", ""))
    else:
        mode = "ir"
        num_layers = int(name.replace("ir_", ""))
    return Backbone(input_size=112, num_layers=num_layers, mode=mode, output_dim=embedding_size)


def load_adaface_pretrained(model, ckpt_path, strict=False):
    """Load AdaFace pretrained .ckpt (Lightning format, 'model.' prefix)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model."):]] = v
        else:
            new_state[k] = v
    result = model.load_state_dict(new_state, strict=strict)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


if __name__ == "__main__":
    for name in ["ir_50", "ir_50_se"]:
        m = build_iresnet(name)
        x = torch.randn(2, 3, 112, 112)
        emb, norm = m(x)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name}: emb={tuple(emb.shape)} norm={tuple(norm.shape)} params={n_params:.2f}M")
