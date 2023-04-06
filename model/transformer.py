import torch
import torch.nn as nn
from einops import rearrange
from model.basic import PreNorm, FeedForward

class MultiHeadAttention(nn.Module):
    """多头注意力层

    Args:
        dim (int): 输入向量的维度
        heads (int, optional): 多头注意力的头数. Defaults to 8.
        dim_head (int, optional): Query, Key, Value的维度. Defaults to 64.
        dropout (float, optional): Dropout的概率. Defaults to 0..
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):      
        b, n, _, h = *x.shape, self.heads
        context = context if context is not None else x

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        # get q k v
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # i,j均表示序列的长度, d是向量的维度
        # 这一行在计算q k之间的点积
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class TransformerEncoder(nn.Module):  
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.):
        """Transformer编码层

        Args:
            dim (int): 输入张量维度
            depth (int): 编码层的深度
            heads (int): 多头注意力的头数
            dim_head (int): 每个头接受张量的维度
            mlp_dim (int): 最后输出张量的维度
            dropout (float, optional): 进行dropout的概率. Defaults to 0..
        """        
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:  # type: ignore
            # 残差连接
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    """张量映射层
    一个张量经过一次映射至符合下一个模块要求的维度，模块的输出再经过一次映射回到张量原来的维度

    Args:
        dim_in (int): 输入向量的维度
        dim_out (int): 需要映射的维度
        fn (function): 映射后向量所要通过的模块
    """
    def __init__(self, dim_in: int, dim_out: int, fn):

        super().__init__()
        self.fn = fn
        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    """交叉Transformer

    Args:
        sm_dim (int): 小分支张量的维度
        lg_dim (int): 大分支张量的维度
        depth (int): 编码层的深度
        heads (int): 多头注意力的头数
        dim_head (int): 每个头接受张量的维度
        mlp_dim (int): 最后输出张量的维度
        dropout (float, optional): 进行dropout的概率
    """
    def __init__(self, sm_dim: int, lg_dim: int, depth: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 这里是将大小两个维度进行交叉Attention
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, MultiHeadAttention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, MultiHeadAttention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # 将CLS与patch分开
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers: # type: ignore
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens


class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        sm_dim: int,
        lg_dim: int,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads: int,
        cross_attn_depth: int,
        cross_attn_dim_head: int = 64,
        dropout: float = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerEncoder(dim = sm_dim, dropout = dropout, **sm_enc_params),
                TransformerEncoder(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers: # type: ignore
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens
    