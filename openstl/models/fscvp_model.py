from torch import nn
from openstl.modules import PatchEmbed, EncoderLayer_S as ES, EncoderLayer_T as ET, DecoderLayer_S as DS, DecoderLayer_T as DT


class FSCVP_Model(nn.Module):
    # def __init__(self, input_shape, hidden_dim, N, NE, ND, patch_size=2, seq_len=10, **kwargs):
    def __init__(self, in_shape, hidden_dim, N, NE, ND, patch_size=2, kernel_size=7, ratio=2, drop_path=.1, init_value=1e-2, seq_len=10, **kwargs):
        super(FSCVP_Model, self).__init__()
        # N must be even(N=number of each layers(embed, enc, dec))
        # T, C, H, W = input_shape
        T, C, H, W = in_shape
        embed = []
        layers = []
        hidden_dims = []
        for i in range(N//2):
            hidden_dims.append(hidden_dim*(2**i))
        hidden_dims = hidden_dims + hidden_dims[::-1]
        
        for i in range(N):
            if i == 0:
                embed.append(PatchEmbed(C, hidden_dims[i], patch_size=patch_size))
            elif i < N//2:
                embed.append(PatchEmbed(hidden_dims[i-1], hidden_dims[i], patch_size=patch_size))
            else:
                embed.append(PatchEmbed(hidden_dims[i-1], hidden_dims[i], patch_size=patch_size, upsampling=True))
        
        for i in range(N):
            if i < N//2:
                enc = []
                for _ in range(NE):
                    # enc.append(ES(dim=hidden_dims[i]))
                    # enc.append(ET(dim=hidden_dims[i]))
                    enc.append(ET(dim=hidden_dims[i], kernel_size=kernel_size, drop_path=drop_path, init_value=init_value))
                    enc.append(ES(dim=hidden_dims[i], kernel_size=kernel_size, drop_path=drop_path, init_value=init_value))
                layers.append(nn.ModuleList(enc))
            else:
                dec = []
                for _ in range(ND):
                    # dec.append(DS(dim=hidden_dims[i]))
                    # dec.append(DT(dim=hidden_dims[i]))
                    dec.append(DT(dim=hidden_dims[i], ratio=ratio))
                    dec.append(DS(dim=hidden_dims[i], ratio=ratio))
                layers.append(nn.ModuleList(dec))

        self.embed = nn.ModuleList(embed)
        self.layers = nn.ModuleList(layers)
        self.finalembed = PatchEmbed(hidden_dim, hidden_dim, patch_size=patch_size, upsampling=True)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], C, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.N = N
        self.NE = NE
        self.ND = ND

    def forward(self, x):
        B, T, C, H, W = x.shape
        enc_attns = []
        idx = 0
        for i in range(self.N):
            if i != self.N//2:
                x = self.embed[i](x)
            if i < self.N//2:
                for j in range(self.NE):
                    x, s_att = self.layers[i][j*2](x)
                    x, t_att = self.layers[i][j*2+1](x)
                enc_attns.append(t_att.clone())
                enc_attns.append(s_att.clone())
            else:
                for j in range(self.ND):
                    s_att = enc_attns[::-1][idx]
                    t_att = enc_attns[::-1][idx+1]
                    x = self.layers[i][j*2](x, s_att)
                    x = self.layers[i][j*2+1](x, t_att)
                idx += 2
        x = self.finalembed(x)
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.final(x)
        x = x.view(B, T, C, H, W)
        return x

