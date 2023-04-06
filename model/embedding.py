import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from model.efficient_net.efficientnet_pytorch import EfficientNet

class ImageEmbedder(nn.Module):
    """图片编码层

    Args:
        dim (int): 输出维度
        image_size (int): 图片大小
        patch_size (int): 把图片分为 `pathc_size ** 2` 个小块，每个小块称为patch
        channels (int): 通道数
        dropout (float, optional): Dropout的概率. Defaults to 0..
        efficient_block (int, optional): 使用的Efficient需要多少个块. Defaults to 8.
    """
    def __init__(
        self,
        *,
        dim: int,
        image_size: int,
        patch_size: int,
        dropout: float = 0.,
        efficient_block: int = 8,
        channels: int
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        # 获取预训练模型
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficient_net.delete_blocks(efficient_block)
        self.efficient_block = efficient_block
        
        # 开启所有参数
        for index, (name, param) in enumerate(self.efficient_net.named_parameters()):
            param.requires_grad = True
        
        # patch这里具体是指将整个图像划分成多个小块，每个小块称为一个patch
        # 这样做的好处是可以将一张大图像转换成多个小向量，从而降低计算复杂度。同时，这种做法也能够使得模型更好地捕捉图像中的局部特征。 
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        '''
        这里解释一下patch_dim为什么这样计算
        在这个模块中，我们将图像划分为大小为patch_size x patch_size的小块，每个小块有channels个通道。因此，每个小块的大小为patch_size x patch_size x channels
        我们希望将每个小块表示为一个向量，因此我们需要将这个三维张量压缩成一个一维向量，这个向量的长度就是patch_dim
        '''
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            # 字符串中的'b', 'c', 'h'和'w'分别代表batch_size、channels、height和width
            # '(h p1)'和'(w p2)'表示将height和width分成大小为patch_size的小块
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        # nn.Parameter是一个特殊的张量类型, 其对象会被自动注册为一个模型参数，并在反向传播时被更新
        '''
        这里解释一下为什么位置编码第二个维度是num_patchs+1
        在这个模块中，我们将整张图像分成大小为patch_size x patch_size的小块，每个小块都被表示为一个向量。因此，图像中共有num_patches个小块。
        在加入位置编码的过程中，我们还需要考虑整张图像本身的位置信息。因此，我们需要为整张图像添加一个特殊的位置编码，用于表示整张图像的位置信息。这个特殊的位置编码被添加在位置编码的第一个位置，因此位置编码的第二个维度大小为num_patches + 1。
        '''
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        '''
        什么是CLS token呢
        是指Transformer中用于表示整个序列的特殊token，通常被称为“CLS”标记。在自然语言处理任务中，这个token通常被用于表示整个句子或文本的语义信息，而在图像处理任务中，它则被用于表示整个图像的特征信息。
        在ViT模型中，我们将整张图像分成大小为patch_size x patch_size的小块，然后将每个小块映射成一个向量表示。这些向量表示被拼接成一个序列，然后在序列的最前面插入一个特殊的CLS标记，表示整个图像的特征信息。
        具体来说，CLS标记的向量表示可以被看作是整张图像的一个紧凑的特征表示，包含了整张图像的全局信息。在图像分类任务中，我们通常将CLS标记的向量表示作为整个图像的分类特征，输入到一个全连接层中进行分类。在图像生成任务中，我们则可以将CLS标记的向量表示作为整个图像的生成向量，用于控制图像的生成过程。
        需要注意的是，CLS标记的向量表示是一个可学习的参数，需要在模型训练过程中不断更新。同时，由于CLS标记的向量表示包含了整张图像的全局信息，因此它对模型的性能有着至关重要的作用。
        '''
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # x的shape应该为(batch_size, channels, feature_size, feature_size)
        x = self.efficient_net.extract_features_at_block(img, self.efficient_block)
        # 分割patch并编码
        x = self.to_patch_embedding(x)
        
        b, n, _ = x.shape
        # 把CLS标志复制batch_size份
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        '''
        之所以在第二个维度进行拼接，是因为我们需要将整张图像的位置编码也加入到序列中
        x:   (batch_size, num_patches, embedding_dim)
        cls: (batch_size, 1, embedding_dim)
        '''
        x = torch.cat((cls_tokens, x), dim=1)
        # 这里的切分实际上没有任何意义，下面这一行代码等价于 x += self.pos_embedding[:, :(n + 1)]
        # self.pos_embedding的shape为 (1, n + 1, embedding_dim)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)
