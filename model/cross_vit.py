import torch.nn as nn
from model.embedding import ImageEmbedder
from model.transformer import MultiScaleEncoder

class CrossEfficientViT(nn.Module):
    def __init__(
        self,
        *,
        config
    ):
        super().__init__()
        image_size = config['model']['image-size']
        num_classes = config['model']['num-classes'] 
        sm_dim = config['model']['sm-dim']
        sm_channels = config['model']['sm-channels']
        lg_dim = config['model']['lg-dim']
        lg_channels = config['model']['lg-channels']         
        sm_patch_size = config['model']['sm-patch-size']
        sm_enc_depth = config['model']['sm-enc-depth'] 
        sm_enc_heads = config['model']['sm-enc-heads']
        sm_enc_mlp_dim = config['model']['sm-enc-mlp-dim']
        sm_enc_dim_head = config['model']['sm-enc-dim-head']
        lg_patch_size = config['model']['lg-patch-size']
        lg_enc_depth = config['model']['lg-enc-depth'] 
        lg_enc_mlp_dim = config['model']['lg-enc-mlp-dim']
        lg_enc_heads = config['model']['lg-enc-heads']
        lg_enc_dim_head = config['model']['lg-enc-dim-head']
        cross_attn_depth = config['model']['cross-attn-depth']
        cross_attn_heads = config['model']['cross-attn-heads']
        cross_attn_dim_head = config['model']['cross-attn-dim-head']
        depth = config['model']['depth']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        # 两个分支的图片Embedder, 两个分支的区别体现在patch_size中
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout, efficient_block = 16, channels=sm_channels)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout, efficient_block = 1, channels=lg_channels)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # 图片编码
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        # 进行计算
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # 获取图片全局信息
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        # 通过多层感知机头部进行预测
        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        # 返回混合信息
        return sm_logits + lg_logits
