import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # 或者其他适合的图卷积层


class ConditionalPositionalEncoding(nn.Module):
    """
    条件位置编码 (CPE) 模块，用于增强图节点的特征表示，使其包含更多空间信息。
    借鉴了 MGC/MobileViGv2 中的 CPE 思想，适用于图结构数据。
    """

    def __init__(self, feature_dim: int, use_residual: bool = True, activation: nn.Module = None):
        """
        初始化 CPE 模块。

        参数:
            feature_dim (int): 输入节点特征的维度，CPE模块的输出维度也将是这个值，
                               以便于进行残差连接。
            use_residual (bool, optional): 是否使用残差连接将 CPE 的输出加回到原始特征。
                                           默认为 True。
            activation (nn.Module, optional): 在卷积后和残差连接前使用的激活函数。
                                             如果为 None，则不使用激活函数。默认为 None。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_residual = use_residual
        self.activation = None
        self.cpe_conv = GCNConv(feature_dim, feature_dim)
        torch.nn.init.xavier_uniform_(self.cpe_conv.lin.weight, gain=0.1) # 使用较小的增益
        if self.cpe_conv.lin.bias is not None:
            torch.nn.init.zeros_(self.cpe_conv.lin.bias)
        # 使用一个简单的图卷积层 (GCNConv) 来学习位置编码。
        # GCNConv 会根据节点的邻域信息来更新节点特征。
        # 输入和输出维度相同，方便残差连接。
        self.cpe_conv = GCNConv(in_channels=feature_dim, out_channels=feature_dim)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            node_feat (torch.Tensor): 节点的输入特征，形状为 (num_nodes, feature_dim)。
                                     这些特征通常是节点当前的位置嵌入。
            edge_index (torch.Tensor): 图的边索引，形状为 (2, num_edges)。

        返回:
            torch.Tensor: 经过 CPE 增强的节点特征，形状与 node_feat 相同。
        """
        # 通过图卷积层生成条件位置信号
        cpe_signal = self.cpe_conv(node_feat, edge_index)

        if self.activation:
            cpe_signal = self.activation(cpe_signal)

        if self.use_residual:
            # 将学习到的位置信号加回到原始特征
            enhanced_feat = node_feat + cpe_signal
        else:
            enhanced_feat = cpe_signal

        return enhanced_feat