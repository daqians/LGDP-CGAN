"""LGDP-CGAN generator: FMS encoder with log conditioning at every scale, multihead log encoder,
and log-gated skip connections in the decoder."""
import functools

import torch
import torch.nn as nn

from .networks import ResnetBlock, get_norm_layer


# =====================================================================
# FiLMLayer - lightweight encoder fusion for shallow layers (E0, E1)
# =====================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = feat * (1 + gamma) + beta"""

    def __init__(self, channels, log_dim):
        super().__init__()
        self.proj = nn.Linear(log_dim, channels * 2)

    def forward(self, feat, log_vec):
        # log_vec: (B, log_dim) -> (B, 2C)
        params = self.proj(log_vec)
        gamma, beta = params.chunk(2, dim=1)
        # broadcast to spatial dims: (B, C) -> (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # Clamp gamma to prevent extreme early-training modulation
        return feat * (1.0 + gamma.clamp(-0.5, 0.5)) + beta


# =====================================================================
# ConcatFusion - full encoder fusion for deep layers (E2, E3)
# =====================================================================

class ConcatFusion(nn.Module):
    """Concat + 1x1 Conv fusion: cat(feat, log_broadcast) -> Conv1x1 -> IN -> ReLU"""

    def __init__(self, channels, norm_layer):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            norm_layer(channels),
            nn.ReLU(True),
        )

    def forward(self, feat, log_spatial):
        # log_spatial already broadcast to (B, C, H, W)
        return self.fuse(torch.cat([feat, log_spatial], dim=1))


# =====================================================================
# AttentionGate - decoder skip gate
# =====================================================================

class AttentionGate(nn.Module):
    """
    Gate = Sigmoid(Conv1x1(ReLU(Conv1x1(cat(skip, log)))))
    Output = skip * (1 + gate)

    The final conv is initialised to zero weight and bias -4, so sigmoid(gate) ~ 0.018 at the start.
    """

    def __init__(self, channels, reduction=4, norm_layer=None):
        super().__init__()
        mid = max(channels // reduction, 1)
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1),
            norm_layer(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, channels, kernel_size=1),
        )
        # Init near-identity: sigmoid(-4) ~ 0.018, gates are essentially closed at start
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, skip, log_spatial):
        g = torch.sigmoid(self.gate(torch.cat([skip, log_spatial], dim=1)))
        return skip * (1.0 + g)


# =====================================================================
# LogMultiScaleProjector - shared trunk + per-scale heads
# =====================================================================

class LogMultiScaleProjector(nn.Module):
    """
    LOG (B,8) -> Shared Trunk (8->256->512) -> (n_downsampling+1) branch heads:
      n_downsampling=3: Head_0..3: 64, 128, 256, 512
      n_downsampling=4: Head_0..4: 64, 128, 256, 512, 1024
    """

    def __init__(self, log_input_nc=8, ngf=64, n_downsampling=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(log_input_nc, 256),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
        )
        # (n_downsampling+1) scale heads: channels = ngf * 2^i
        self.heads = nn.ModuleList()
        for i in range(n_downsampling + 1):
            ch = ngf * (2 ** i)
            self.heads.append(nn.Sequential(
                nn.Linear(512, ch),
                nn.LayerNorm(ch),
                nn.ReLU(True),
            ))

    def forward(self, log_input):
        """Returns list of (n_downsampling+1) vectors"""
        shared = self.trunk(log_input)
        return [head(shared) for head in self.heads]


# =====================================================================
# LGDPGenerator
# =====================================================================

class LGDPGenerator(nn.Module):
    """
    Full-chain multi-modal conditioned generator. Supports n_downsampling=3 or 4.

    Encoder (n=3): initial_conv->[FiLM_0]->skip_0->down->[FiLM_1]->skip_1
                   ->down->[ConcatFusion_2]->skip_2->down->[ConcatFusion_3/bottleneck]
    Encoder (n=4): adds one more ConcatFusion_4/bottleneck level

    Bottleneck:  9x ResNet blocks

    Decoder: n upsample steps, each with attention-gated skip connection
    """

    def __init__(self, fms_input_nc=1, log_input_nc=8, output_nc=3, ngf=64,
                 n_downsampling=3, n_blocks=9, norm='instance',
                 padding_type='reflect', gate_reduction=4):
        super().__init__()
        self.n_downsampling = n_downsampling
        norm_layer = get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)

        # ---- LOG multi-scale projector ----
        self.log_projector = LogMultiScaleProjector(log_input_nc, ngf, n_downsampling)

        # ---- FMS encoder: initial conv ----
        self.initial_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(fms_input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        )

        # ---- FMS encoder: downsample blocks ----
        self.downsamples = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            self.downsamples.append(nn.Sequential(
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation,
            ))

        # ---- Encoder fusion modules ----
        # E0, E1: FiLM (lightweight); E2..n_downsampling: ConcatFusion
        self.enc_fusions = nn.ModuleList()
        for i in range(n_downsampling + 1):
            ch = ngf * (2 ** i)
            if i < 2:
                self.enc_fusions.append(FiLMLayer(ch, ch))
            else:
                self.enc_fusions.append(ConcatFusion(ch, norm_layer))

        # ---- ResNet blocks ----
        bottleneck_ch = ngf * (2 ** n_downsampling)  # n=3: 512, n=4: 1024
        resnet_layers = []
        for _ in range(n_blocks):
            resnet_layers.append(
                ResnetBlock(bottleneck_ch, padding_type=padding_type,
                            activation=activation, norm_layer=norm_layer)
            )
        self.resnet_blocks = nn.Sequential(*resnet_layers)

        # ---- Decoder: upsample + refine ----
        self.upsamples = nn.ModuleList()
        self.refine_convs = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            out_ch = int(ngf * mult / 2)
            self.upsamples.append(nn.Sequential(
                nn.ConvTranspose2d(ngf * mult, out_ch,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(out_ch),
                activation,
            ))
            self.refine_convs.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                norm_layer(out_ch),
                activation,
            ))

        # ---- Attention gates for skip connections ----
        # AG_2 (256ch), AG_1 (128ch), AG_0 (64ch)
        self.attention_gates = nn.ModuleList()
        for i in range(n_downsampling):
            ch = ngf * (2 ** i)
            self.attention_gates.append(
                AttentionGate(ch, reduction=gate_reduction, norm_layer=norm_layer)
            )

        # ---- Output layer ----
        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, fms_input, log_input):
        """
        Args:
            fms_input: (B, 1, 512, 512)
            log_input: (B, 8)
        Returns:
            output: (B, 3, 512, 512)
        """
        # -- Multi-scale LOG projections --
        log_vecs = self.log_projector(log_input)  # n+1 vecs: [(B,64)..(B, ngf*2^n)]

        # -- Encoder --
        x = self.initial_conv(fms_input)  # (B,64,512,512)

        skips = []
        for i in range(self.n_downsampling):
            # Fusion at current scale
            if i < 2:
                # FiLM: pass log vector directly
                x = self.enc_fusions[i](x, log_vecs[i])
            else:
                # ConcatFusion: need spatial broadcast
                B, C = log_vecs[i].shape
                _, _, H, W = x.shape
                log_sp = log_vecs[i].view(B, C, 1, 1).expand(B, C, H, W)
                x = self.enc_fusions[i](x, log_sp)

            skips.append(x)  # save skip after fusion
            x = self.downsamples[i](x)  # downsample

        # Bottleneck fusion (ConcatFusion at deepest level)
        B, C = log_vecs[self.n_downsampling].shape
        _, _, H, W = x.shape
        log_sp = log_vecs[self.n_downsampling].view(B, C, 1, 1).expand(B, C, H, W)
        x = self.enc_fusions[self.n_downsampling](x, log_sp)

        # -- ResNet blocks --
        x = self.resnet_blocks(x)

        # -- Decoder with attention-gated skips --
        # skips: [skip_0 (64ch), skip_1 (128ch), skip_2 (256ch)]
        # upsamples go: 512->256, 256->128, 128->64
        # AG order: AG_2 uses skip_2, AG_1 uses skip_1, AG_0 uses skip_0
        for i in range(self.n_downsampling):
            x = self.upsamples[i](x)

            skip_idx = self.n_downsampling - 1 - i  # 2, 1, 0
            skip = skips[skip_idx]

            # Attention gate with LOG conditioning
            B_s, C_s = log_vecs[skip_idx].shape
            _, _, H_s, W_s = skip.shape
            log_sp = log_vecs[skip_idx].view(B_s, C_s, 1, 1).expand(B_s, C_s, H_s, W_s)
            gated_skip = self.attention_gates[skip_idx](skip, log_sp)

            x = x + gated_skip
            x = self.refine_convs[i](x)

        # -- Output --
        return self.output_conv(x)




def define_generator(fms_input_nc=1, log_input_nc=8, output_nc=3, ngf=64, n_downsample_global=4,
                     n_blocks_global=9, norm='instance', gate_reduction=4, gpu_ids=()):
    netG = LGDPGenerator(fms_input_nc=fms_input_nc, log_input_nc=log_input_nc, output_nc=output_nc,
                         ngf=ngf, n_downsampling=n_downsample_global, n_blocks=n_blocks_global,
                         norm=norm, gate_reduction=gate_reduction)
    print('Generator [LGDPGenerator]: %s parameters' % format(sum(p.numel() for p in netG.parameters()), ','))
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG.apply(weights_init)
    # attention gates start almost closed (overrides weights_init)
    for ag in netG.attention_gates:
        nn.init.zeros_(ag.gate[-1].weight)
        nn.init.constant_(ag.gate[-1].bias, -4.0)
    return netG
