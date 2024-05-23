# Models

import math

import numpy as np

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block


""" Video-UNETR Model """


class VideoUnetr(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        out_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        skip_chans=[64, 128, 256],
    ):
        super().__init__()  # Default ViT encoder initialization parameters are for ViT-B/16
        if depth % 4 != 0:
            raise ValueError("Depth must be divisible by 4 for skip connections in UNETR.")
        # --------------------------------------------------------------------------
        # ViT encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding in later initialization

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # skip connection blocks
        self.skip_encoder_ids = [
            depth // 4 - 1,
            depth // 4 * 2 - 1,
            depth // 4 * 3 - 1,
        ]  # [2, 5, 8] for depth=12 & [1, 3, 5] for depth=8
        self.skip1 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[0]),
            SkipBlock(skip_chans[0], skip_chans[0]),
            SkipBlock(skip_chans[0], skip_chans[0]),
        )
        self.skip2 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[1]),
            SkipBlock(skip_chans[1], skip_chans[1]),
        )
        self.skip3 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[2]),
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # upsampling blocks
        self.bottem_up = UpsampleBlock(embed_dim, skip_chans[2])
        self.up3 = UpsampleBlock(skip_chans[2], skip_chans[1])
        self.up2 = UpsampleBlock(skip_chans[1], skip_chans[0])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(skip_chans[0], skip_chans[0] // 2, kernel_size=4, stride=2, padding=1),
            ResidualBlock(skip_chans[0] // 2, skip_chans[0] // 2),
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # output
        self.out_conv = nn.Conv2d(skip_chans[0] // 2, out_chans, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    # def unpatchify(self, x): # for MAE pre-training only
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    def vit_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)  # [N, L, D]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # [N, L, D]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # [1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [N, L+1, D]

        # store latent features from ViT block
        latents_out = []

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            latents_out.append(x)
        x = self.norm(x)

        return x, latents_out

    def latent_reshape(self, latents):
        """
        input:
        latents: (N, L+1, D)
        return:
        latents: (N, D, H//p, W//p)
        """
        # remove cls token
        latents = latents[:, 1:, :]

        h = w = int(latents.shape[1] ** 0.5)
        latents = latents.reshape(shape=(latents.shape[0], h, w, latents.shape[2]))  # [N, H//p, W//p, D]

        # reshape to (N, D, H//p, W//p)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    def forward(self, imgs):
        # ViT encoder
        x, latents_out = self.vit_encoder(imgs)  # x & all latents with dim [N, L+1, D]

        # default skip_chans: [64, 128, 256]
        # skip connection from latents 1
        latents1 = self.latent_reshape(latents_out[self.skip_encoder_ids[0]])  # [N, D, H//p, W//p]
        s1 = self.skip1(latents1)  # [N, skip_chans[0], H//2, W//2]

        # skip connection from latents 2
        latents2 = self.latent_reshape(latents_out[self.skip_encoder_ids[1]])  # [N, D, H//p, W//p]
        s2 = self.skip2(latents2)  # [N, skip_chans[1], H//4, W//4]

        # skip connection from latents 3
        latents3 = self.latent_reshape(latents_out[self.skip_encoder_ids[2]])  # [N, D, H//p, W//p]
        s3 = self.skip3(latents3)  # [N, skip_chans[2], H//8, W//8]

        # the last feature output from ViT encoder
        x = self.latent_reshape(x)  # [N, D, H//p, W//p]

        # upsampling blocks
        x = self.bottem_up(x, s3)  # [N, skip_chans[2], H//8, W//8]
        x = self.up3(x, s2)  # [N, skip_chans[1], H//4, W//4]
        x = self.up2(x, s1)  # [N, skip_chans[0], H//2, W//2]
        x = self.up1(x)  # [N, skip_chans[0]//2, H, W]

        # output
        x = self.out_conv(x)  # [N, out_chans, H, W]
        x = self.sigmoid(x)

        return x


""" Video-Retina-UNETR Model """


class VideoRetinaUnetr(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        out_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        skip_chans=[64, 128, 256],
        num_anchors=9,
        num_classes=2,
        det_feature_size=256,
    ):
        super().__init__()  # Default ViT encoder initialization parameters are for ViT-B/16
        if depth % 4 != 0:
            raise ValueError("Depth must be divisible by 4 for skip connections in UNETR.")
        # --------------------------------------------------------------------------
        # ViT encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding in later initialization

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # skip connection blocks
        self.skip_encoder_ids = [
            depth // 4 - 1,
            depth // 4 * 2 - 1,
            depth // 4 * 3 - 1,
        ]  # [2, 5, 8] for depth=12 & [1, 3, 5] for depth=8
        self.skip1 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[0]),
            SkipBlock(skip_chans[0], skip_chans[0]),
            SkipBlock(skip_chans[0], skip_chans[0]),
        )
        self.skip2 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[1]),
            SkipBlock(skip_chans[1], skip_chans[1]),
        )
        self.skip3 = nn.Sequential(
            SkipBlock(embed_dim, skip_chans[2]),
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # retina head modules
        self.pyramid_features = PyramidFeatures(skip_chans[2], embed_dim, det_feature_size)
        self.cls_head = ClassificationHead(det_feature_size, num_anchors, num_classes, det_feature_size)
        self.reg_head = RegressionHead(det_feature_size, num_anchors, det_feature_size)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # upsampling blocks
        self.bottem_up = UpsampleBlock(embed_dim, skip_chans[2])
        self.up3 = UpsampleBlock(skip_chans[2], skip_chans[1])
        self.up2 = UpsampleBlock(skip_chans[1], skip_chans[0])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(skip_chans[0], skip_chans[0] // 2, kernel_size=4, stride=2, padding=1),
            ResidualBlock(skip_chans[0] // 2, skip_chans[0] // 2),
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # output
        self.out_conv = nn.Conv2d(skip_chans[0] // 2, out_chans, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize classification head and regression head
        prior = 0.01
        self.cls_head.output.weight.data.fill_(0)
        self.cls_head.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.reg_head.output.weight.data.fill_(0)
        self.reg_head.output.bias.data.fill_(0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    # def unpatchify(self, x): # for MAE pre-training only
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    def vit_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)  # [N, L, D]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # [N, L, D]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # [1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # [N, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [N, L+1, D]

        # store latent features from ViT block
        latents_out = []

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            latents_out.append(x)
        x = self.norm(x)

        return x, latents_out

    def latent_reshape(self, latents):
        """
        input:
        latents: (N, L+1, D)
        return:
        latents: (N, D, H//p, W//p)
        """
        # remove cls token
        latents = latents[:, 1:, :]

        h = w = int(latents.shape[1] ** 0.5)
        latents = latents.reshape(shape=(latents.shape[0], h, w, latents.shape[2]))  # [N, H//p, W//p, D]

        # reshape to (N, D, H//p, W//p)
        latents = latents.permute(0, 3, 1, 2)

        return latents

    def forward(self, imgs):
        # ViT encoder
        x, latents_out = self.vit_encoder(imgs)  # x & all latents with dim [N, L+1, D]

        # default skip_chans: [64, 128, 256]
        # skip connection from latents 1
        latents1 = self.latent_reshape(latents_out[self.skip_encoder_ids[0]])  # [N, D, H//p, W//p]
        s1 = self.skip1(latents1)  # [N, skip_chans[0], H//2, W//2]

        # skip connection from latents 2
        latents2 = self.latent_reshape(latents_out[self.skip_encoder_ids[1]])  # [N, D, H//p, W//p]
        s2 = self.skip2(latents2)  # [N, skip_chans[1], H//4, W//4]

        # skip connection from latents 3
        latents3 = self.latent_reshape(latents_out[self.skip_encoder_ids[2]])  # [N, D, H//p, W//p]
        s3 = self.skip3(latents3)  # [N, skip_chans[2], H//8, W//8]

        # the last feature output from ViT encoder
        x = self.latent_reshape(x)  # [N, D, H//p, W//p]

        # feature pyramid extraction block
        features = self.pyramid_features(s3, x)

        # retina head
        classifications = torch.cat([self.cls_head(f) for f in features], dim=1)  # [N, num_total_anchors, num_classes]
        regressions = torch.cat([self.reg_head(f) for f in features], dim=1)  # [N, num_total_anchors, 4]

        # upsampling blocks
        x = self.bottem_up(x, s3)  # [N, skip_chans[2], H//8, W//8]
        x = self.up3(x, s2)  # [N, skip_chans[1], H//4, W//4]
        x = self.up2(x, s1)  # [N, skip_chans[0], H//2, W//2]
        x = self.up1(x)  # [N, skip_chans[0]//2, H, W]

        # output
        x = self.out_conv(x)  # [N, out_chans, H, W]
        x = self.sigmoid(x)

        return x, classifications, regressions


""" Positional Encoding for ViT """


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


""" Blocks for Skip Connections & Upsampling """


# single convolutional block with GroupNorm and GELU activation (single yellow block in the architecture diagram)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.dropout.p > 0.0:
            x = self.dropout(x)
        return x


# 2 BasicBlock with residual connection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = BasicBlock(in_channels, out_channels)
        self.block2 = BasicBlock(out_channels, out_channels)
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out + self.res(x)
        return out


# skip connection block (transpose convolution followed by a BasicBlock. blue block in the architecture diagram)
class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transp = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.transp(x)
        x = self.block(x)
        return x


# upsampling block (green block followed by concatenation and 2 yellow blocks in the architecture diagram)
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


""" Blocks for Retina Head """
# https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/model.py


# feature pyramid extraction block
class PyramidFeatures(nn.Module):
    def __init__(self, skip3_channels, bottom_channels, feature_channels=256):
        super().__init__()

        # bottom features convolutions and upsampling
        self.bottom_conv1 = nn.Conv2d(bottom_channels, feature_channels, kernel_size=1, stride=1, padding=0)
        self.bottom_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.bottom_conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1)

        # latents 3 convolutions
        self.s3_conv1 = nn.Conv2d(skip3_channels, feature_channels, kernel_size=1, stride=1, padding=0)
        self.s3_conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1)

        # bottem features downsampling via 3x3 conv with stride 2
        self.bottom_down_1 = nn.Conv2d(bottom_channels, feature_channels, kernel_size=3, stride=2, padding=1)

        # bottom features downsampling again via 3x3 conv with stride 2
        self.bottom_down_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, skip3, bottom):
        # bottom features
        bottom_x = self.bottom_conv1(bottom)
        bottom_up_x = self.bottom_up(bottom_x)
        bottom_x = self.bottom_conv2(bottom_x)

        # skip 3 features
        skip3_x = self.s3_conv1(skip3)
        skip3_x = skip3_x + bottom_up_x
        skip3_x = self.s3_conv2(skip3_x)

        # bottom features after 1 downsampling
        bottom_down_1_x = self.bottom_down_1(bottom)

        # bottom features after 2 downsampling
        bottom_down_2_x = self.bottom_down_2(bottom_down_1_x)

        return skip3_x, bottom_x, bottom_down_1_x, bottom_down_2_x


class ClassificationHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=1, feature_channels=256):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv = nn.Sequential(
            nn.Conv2d(num_features_in, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(feature_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.output(x)
        x = self.output_act(x)
        x = x.permute(0, 2, 3, 1)  # B x H x W x (num_anchors * num_classes)
        x = x.reshape(x.shape[0], -1, self.num_classes)  # B x (W x H x num_anchors) x num_classes

        return x


class RegressionHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_channels=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_features_in, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(feature_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)  # B x H x W x (num_anchors * 4)
        x = x.reshape(x.shape[0], -1, 4)  # B x (H x W x num_anchors) x 4

        # apply sigmoid with shift of -0.5 and scale of pi/2 to the angle output
        x[:, :, 2] = (self.sigmoid(x[:, :, 2]) - 0.5) * math.pi  ## TODO: maybe modify to a flatten version of the sigmoid

        return x
