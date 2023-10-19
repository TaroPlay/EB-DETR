import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay

__all__ = ['HybridEncoder']

    
'''-----------------SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Layer):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)

#CCFM模块中的fusion block
class CSPRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        
        # 连续3个VggBlock
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

#transformerLayer是论文中AIFI的模块,暂不修改
@register
class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

#主要内容在下面
@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        # 路由融合迭代次数,默认为2
        self.times = 2

        # channel projection
        #   input_proj包含3个network: LayerList(
        #   (0): Sequential(
        #     (0): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
        #     (1): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        #   )
        #   (1): Sequential(
        #     (0): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
        #     (1): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        #   )
        #   (2): Sequential(
        #     (0): Conv2D(2048, 256, kernel_size=[1, 1], data_format=NCHW)
        #     (1): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        #   )
        # )
        # 把通道数修改成256对齐
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))


        # encoder transformer
        # AIFI模块 实验证明不需要
        # self.encoder = nn.LayerList([
        #     TransformerEncoder(encoder_layer, num_encoder_layers)
        #     for _ in range(len(use_encoder_idx))
        # ])

        # 激活函数
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))
    
        # 动态路由部分fpn和pan 两层，一层3个模块
        # self.pan_blocks_route = nn.LayerList()
        # self.fpn_blocks_route = nn.LayerList()
        # #idx = 0 , idx = 1 ,idx = 2 
        # for idx in range(3):
        #     self.pan_blocks_route.append(
        #         CSPRepLayer(
        #             hidden_dim * 2,
        #             hidden_dim,
        #             round(3 * depth_mult),
        #             act=act,
        #             expansion=expansion))
        #     self.fpn_blocks_route.append(
        #         CSPRepLayer(
        #             hidden_dim * 2,
        #             hidden_dim,
        #             round(3 * depth_mult),
        #             act=act,
        #             expansion=expansion))

        # # 动态路由部分lateral和downsample 
        # # S3 -> S3' 下采样一次 ; F5' -> F5'' 下采样一次
        # # F5 -> F5' 1*1 conv ; F5'到F5'' 1*1 conv
        # self.downsample_convs_route = nn.LayerList()
        # self.lateral_convs_route = nn.LayerList()
        # #idx = 0 , idx = 1
        # for idx in range(2):
        #     self.downsample_convs_route.append(
        #         BaseConv(
        #             hidden_dim, hidden_dim, 3, stride=2, act=act))
        #     self.lateral_convs_route.append(
        #         BaseConv(
        #             hidden_dim, hidden_dim, 1, 1, act=act))
            

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        #feats:[8,512,68,68] [8,1024,34,34] [8,2048,17,17]对应S3，S4，S5
        assert len(feats) == len(self.in_channels)
        
        # get projection features：S3,S4,S5的变化只是通道数全部变为256
        # proj_feats[8,256,68,68],[8,256,34,34],[8,256,17,17]
        # proj_feats[8,256,80,80],[8,256,40,40],[8,256,20,20]
        # 图片size不同，最后两个维度会有不同
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # AIFI encoder：使用一个transformer块,并且只把S5去做自交互
        # 实验证明不用transformer更好
        # if self.num_encoder_layers > 0:
        #     for i, enc_ind in enumerate(self.use_encoder_idx):
        #         h, w = proj_feats[enc_ind].shape[2:]
        #         # flatten [B, C, H, W] to [B, HxW, C]
        #         src_flatten = proj_feats[enc_ind].flatten(2).transpose(
        #             [0, 2, 1])
        #         if self.training or self.eval_size is None:
        #             pos_embed = self.build_2d_sincos_position_embedding(
        #                 w, h, self.hidden_dim, self.pe_temperature)
        #         else:
        #             pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
        #         memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
        #         proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
        #             [-1, self.hidden_dim, h, w])
        
        # route = proj_feats[:]
        # # dynamic
        # # 创新点修改：对proj_feats[0],proj_feats[1],proj_feats[2]进行路由连接方式的融合
        # for idx in range(self.times):
        #     #获取S3,S4,F5
        #     feat_low = route[0]
        #     feat_middle = route[1]
        #     feat_heigh = route[2]
        #     #F5通过1×1卷积
            
        #     #upsample_feat 等价于 feat_heigh F5
        #     #downsample_feat 等价于 feat_low S3
        #     #feat_middle 等价于 S4
        #     if idx == 0:
        #         #F5 做 1×1 conv
        #         feat_heigh = self.lateral_convs_route[0](feat_heigh)
        #         #F5 上采样
        #         upsample_feat = F.interpolate(
        #             feat_heigh, scale_factor=2., mode="nearest")
        #         #S3 下采样
        #         downsample_feat = self.downsample_convs_route[0](feat_low)
                
        #         #fpn的路由链接
        #         route[0] = self.fpn_blocks_route[0](
        #             paddle.concat(
        #                 [upsample_feat, feat_middle], axis=1))
        #         route[1] = self.fpn_blocks_route[1](
        #             paddle.concat(
        #                 [upsample_feat, downsample_feat], axis=1))
        #         route[2] = self.fpn_blocks_route[2](
        #             paddle.concat(
        #                 [feat_middle, downsample_feat], axis=1))
        #     elif idx == 1:
        #         #F5' 做 1×1 conv
        #         feat_heigh = self.lateral_convs_route[1](feat_heigh)
        #         #pan的路由链接
        #         route[0] = self.pan_blocks_route[0](paddle.concat(
        #             [feat_heigh, feat_middle], axis=1))
        #         route[1] = self.pan_blocks_route[1](paddle.concat(
        #             [feat_heigh, feat_low], axis=1))
        #         route[2] = self.pan_blocks_route[2](paddle.concat(
        #             [feat_middle, feat_low], axis=1))
            
        # route_outs = []
        # #F5'' 做下采样
        # route_outs.insert(0,self.downsample_convs_route[1](route[0]))
        # route_outs.insert(0,route[1])
        # route_outs.insert(0,F.interpolate(route[2], scale_factor=2., mode="nearest"))
        #end of dynamic_route

        #
        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)
        

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)
        # total是 outs和route相加
        # total = []
        # for i in range(len(route_outs)):
        #     total.append(route_outs[i] + outs[i])

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]
