import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from einops.layers.torch import Rearrange
MIN_NUM_PATCHES = 16


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT_Encoder(nn.Module):#定义基于transformer的编码器
    def __init__(self, *, image_size, patch_size, dim=1024, depth, heads, mlp_dim, pool = 'cls', channels = 200, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT_Encoder,self).__init__()

        self.patch_size = patch_size
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        # num patches -> (224 / 16) = 14, 14 * 14 = 196
        num_patches = (image_size // patch_size) ** 2

        # path dim -> 3 * 16 * 16 = 768，和Bert-base一致
        patch_dim = channels * patch_size ** 2



        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 输出选cls token还是做平均池化

        # 步骤一：图像分块与映射。首先将图片分块，然后接一个线性层做映射
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        # pos_embedding：位置编码；可学习编码，MAE中使用绝对位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # cls_token：在序列最前面插入一个cls token作为分类输出
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 步骤二：Transformer Encoder结构来提特征
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool#类初始化这里定义为'cls',这里我们可以改为'mean'
        self.to_latent = nn.Identity()  # 不区分参数的占位符标识运算符。意思就是这个网络层的设计是用于占位的，即不干活，只是有这么一个层，放到残差网络里就是在跳过连接的地方用这个层，显得没有那么空虚！

        self.mu=nn.Linear(dim,dim)#为每个patch学一个均值
        self.log_sigma=nn.Linear(dim,dim)#为每个patch学一个方差
        
          
    def forward(self, img):
       
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        
        x += self.pos_embedding[:, :(n)]

        x = self.dropout(x)

        x = self.transformer(x)#(32,25,2048)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]#(32,2048)，只取第一个token，既cls-token,用于分类

        x = self.to_latent(x)#(32,25,2048)

        # ！！！！！！！在这里学一个均值和方差
        H = x.mean(dim = 1)#求这25个token的均值

        mu=self.mu(H)
        log_sigma=self.log_sigma(H)
        sigma=torch.exp(log_sigma)

        return H,mu,sigma
 

class ViT_Decoder(nn.Module):#定义基于transformer的解码器
    def __init__(self, *, image_size,patch_size,dim=900, depth, heads, mlp_dim, pool = 'cls', channels = 200, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT_Decoder,self).__init__()
        self.patch_size = patch_size    #3*3
        patch_dim = channels * patch_size ** 2  #patch_dim = 1800
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(dim, patch_dim),
        )

    def forward(self, x):

        x = self.transformer(x)#(32,26,2048),，32为batch-size,26指的是1+25，一个cls-token和25个patch,2048为表征维度
        x = self.to_patch_embedding(x)
        x = self.to_latent(x)

        return x

        