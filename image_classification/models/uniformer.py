import math
from collections import OrderedDict
from functools import partial

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.module import GELU, LayerNorm

from .utils import DropPath, to_2tuple, trunc_normal, TanH
from .vision_transformer import Attention, Mlp, LayerScale, PatchEmbed

CMlp = partial(Mlp, fc_type=partial(M.Conv2d, kernel_size=1))

layer_scale = False
init_value = 1e-6

class CBlock(M.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=GELU,
        norm_layer=LayerNorm,
    ):
        super(CBlock, self).__init__()
        self.pos_embed = M.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = M.BatchNorm2d(dim)
        self.conv1 = M.Conv2d(dim, dim, 1)
        self.conv2 = M.Conv2d(dim, dim, 1)
        self.attn = M.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else M.Identity()
        self.norm2 = M.BatchNorm2d(dim)
        self.mlp = CMlp(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            act_layer=act_layer, 
            drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + \
            self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(M.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        act_layer=GELU, 
        norm_layer=LayerNorm,
    ):
        super(SABlock, self).__init__()
        self.pos_embed = M.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else M.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, 
            drop=drop)
        global layer_scale
        if layer_scale:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = LayerScale(dim, init_values=init_value)
            self.gamma_2 = LayerScale(dim, init_values=init_value)
        else:
            self.gamma_1 = M.Identity()
            self.gamma_2 = M.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = F.flatten(x, 2).transpose((0, 2, 1))
        x = x + self.drop_path(self.gamma_1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.gamma_2(self.mlp(self.norm2(x))))
        x = x.transpose((0, 2, 1)).reshape(B, N, H, W)
        return x


class head_embedding(M.Sequential):
    def __init__(self, in_channels, out_channels):
        proj = [
            M.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels // 2,
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
            M.BatchNorm2d(out_channels // 2),
            M.GELU(),
            M.Conv2d(
                in_channels=out_channels // 2, 
                out_channels=out_channels,
                kernel_size=3, 
                stride=2, 
                padding=1
            ),
            M.BatchNorm2d(out_channels),
        ]
        super(head_embedding, self).__init__(proj)


class middle_embedding(M.Sequential):
    def __init__(self, in_channels, out_channels):

        proj = [
            M.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            M.BatchNorm2d(out_channels),
        ]
        super(middle_embedding, self).__init__()


class UniFormer(M.Module):
    def __init__(
        self, 
        depth=[3, 4, 8, 3], 
        img_size=224, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=[64, 128, 320, 512],
        head_dim=64, 
        mlp_ratio=4., 
        qkv_bias=True, 
        representation_size=None,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=partial(M.LayerNorm, eps=1e-6),
        conv_stem=False,
    ):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (M.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        if conv_stem:
            self.patch_embed1 = head_embedding(
                in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = middle_embedding(
                in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(
                in_channels=embed_dim[1], out_channels=embed_dim[2])
            self.patch_embed4 = middle_embedding(
                in_channels=embed_dim[2], out_channels=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, 
                patch_size=4, 
                in_chans=in_chans, 
                embed_dim=embed_dim[0], 
                flatten=False)
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, 
                patch_size=2, 
                in_chans=embed_dim[0], 
                embed_dim=embed_dim[1],
                 flatten=False)
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, 
                patch_size=2, 
                in_chans=embed_dim[1],
                embed_dim=embed_dim[2], 
                flatten=False)
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, 
                patch_size=2, 
                in_chans=embed_dim[2], 
                embed_dim=embed_dim[3], 
                flatten=False)

        self.pos_drop = M.Dropout(drop_rate)
        dpr = [x.item() for x in F.linspace(0, drop_path_rate,
                                                sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = [
            CBlock(
                dim=embed_dim[0], 
                num_heads=num_heads[0], 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer
            )
            for i in range(depth[0])
        ]
        self.blocks2 = [
            CBlock(
                dim=embed_dim[1], 
                num_heads=num_heads[1], 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i+depth[0]], 
                norm_layer=norm_layer
            )
            for i in range(depth[1])
        ]
        self.blocks3 = [
            SABlock(
                dim=embed_dim[2], 
                num_heads=num_heads[2], 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i+sum(depth[0:2])], 
                norm_layer=norm_layer
            )
            for i in range(depth[2])
        ]
        self.blocks4 = [
            SABlock(
                dim=embed_dim[3], 
                num_heads=num_heads[3], 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i+sum(depth[0:3])], 
                norm_layer=norm_layer
            )
            for i in range(depth[3])
        ]
        self.norm = M.BatchNorm2d(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = M.Sequential(OrderedDict([
                ('fc', M.Linear(embed_dim, representation_size)),
                ('act', TanH())
            ]))
        else:
            self.pre_logits = M.Identity()

        # Classifier head
        self.head = M.Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else M.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, M.Linear):
            m.weight = trunc_normal(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.zeros_(m.bias)
        elif isinstance(m, M.LayerNorm):
            M.init.zeros_(m.bias)
            M.init.ones_(m.weight)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = M.Linear(self.embed_dim, num_classes) if num_classes > 0 else M.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.flatten(x, 2).mean(-1)
        x = self.head(x)
        return x


def uniformer_small(**kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def uniformer_small_plus(**kwargs):
    model = UniFormer(
        depth=[3, 5, 9, 3], conv_stem=True,
        embed_dim=[64, 128, 320, 512], head_dim=32, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def uniformer_small_plus_dim64(**kwargs):
    model = UniFormer(
        depth=[3, 5, 9, 3], conv_stem=True,
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def uniformer_base(**kwargs):
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model


def uniformer_base_ls(**kwargs):
    global layer_scale
    layer_scale = True
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(M.LayerNorm, eps=1e-6), **kwargs)
    return model
