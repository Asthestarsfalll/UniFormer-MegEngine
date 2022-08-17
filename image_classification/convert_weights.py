import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn

from models.torch_model import uniformer_base as torch_uniformer_base
from models.torch_model import uniformer_base_ls as torch_uniformer_base_ls
from models.torch_model import uniformer_small as torch_uniformer_small
from models.torch_model import \
    uniformer_small_plus as torch_uniformer_small_plus
from models.torch_model import \
    uniformer_small_plus_dim64 as torch_uniformer_small_plus_dim64
from models.uniformer import (uniformer_base, uniformer_base_ls,
                              uniformer_small, uniformer_small_plus,
                              uniformer_small_plus_dim64)


def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    conv_stem = torch_model.conv_stem
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        if conv_stem and "patch_embed" in k:
            print(k, " -> ", k.replace('proj.', ''), "with shape: ", data.shape)
            new_dict[k.replace('proj.', '')] = data
        else:
            new_dict[k] = data
    return new_dict


def main(torch_name, torch_path):
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    if torch_name in ["uniformer_small", "uniformer_small_plus_dim64"]:
        torch_state_dict = torch_state_dict['model']
    torch_model = eval("torch_" + torch_name)()
    torch_model.load_state_dict(torch_state_dict)

    new_dict = convert(torch_model, torch_state_dict)
    model = eval(torch_name)()

    error = model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='uniformer_base',
        help=f"Path to torch saved model, default: uniformer_base, optional: uniformer_base_ls, uniformer_small, uniformer_small_plus, uniformer_small_plus_dim64",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default=None,
        help=f"Path to torch saved model, default: None",
    )
    args = parser.parse_args()
    main(args.model, args.dir)
