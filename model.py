import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from base import BaseModel
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max
import torch.sparse as sparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) # 创建weight
        self.init_params()                                                # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)   # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class Classification(nn.Module):                                         # 把GraphSAGE的输出链接全连接层每个节点映射到7维
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)


class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter((torch.FloatTensor(out_features)))

        else:
            self.register_parameter('bias',None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,input,adj):
        support = torch.mm(input,self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output+self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+"->"+str(self.out_features)+')'
class GCN(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout):
        super(GCN,self).__init__()
        self.gc1 = GraphConvolution(nfeat,nhid)
        self.gc2 = GraphConvolution(nhid,nclass)
        """"""
        self.residual = nn.Linear(nfeat,nclass)
        """"""
        self.dropout = dropout

    def forward(self,x,adj):
        residual = self.residual(x)
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x,self.dropout,training=self.training)
        # x = self.residual(x)
        x = F.relu(self.gc2(x,adj)+residual)
        return F.log_softmax(x,dim=1)


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#
#         # 定义两个卷积层和批归一化层
#         self.conv1 = nn.Conv1d(in_channels=in_channels,
#                                out_channels=out_channels,
#                                kernel_size=3,
#                                padding=1)
#
#         self.bn1 = nn.BatchNorm1d(out_channels)
#
#         self.conv2 = nn.Conv1d(in_channels=out_channels,
#                                out_channels=out_channels,
#                                kernel_size=3,
#                                padding=1)
#
#         self.bn2 = nn.BatchNorm1d(out_channels)
#
#     def forward(self, x):
#         # 保存输入信号x到identity变量中
#         identity = x
#
#         # 对x进行一系列卷积、批归一化和ReLU操作
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
# # 如果输入和输出的形状不同，则需要进行降采样或升采样操作，并补充零以使它们具有相同的形状
#         if identity.shape[2] != x.shape[2]:
#             identity = F.avg_pool1d(identity, kernel_size=3, stride=2, padding=0)  # 降采样
#
#             # 补零操作，使得identity和x的channel数相同
#             ch1 = identity.shape[1]
#             ch2 = x.shape[1]
#             zeros = torch.zeros((x.shape[0], ch1 - ch2, x.shape[2]), dtype=torch.float32).to(x.device)
#             x = torch.cat([x, zeros], dim=1)
#
#         # 将identity和x相加，并通过ReLU激活函数输出
#         x += identity
#         x = F.relu(x)
#
#         return x


import math
import numpy as np
import time
import os
import config

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max
import torch.sparse as sparse

class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, out_dim, dropout, alpha, concat=True, residual=False):
        super(SparseGATLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.residual = residual
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        device = x.device
        N = x.size()[0]   # 图中节点数
        edge = adj._indices()   # 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        if x.is_sparse:   # 判断特征是否为稀疏矩阵
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        values = self.attn.mm(edge_h).squeeze()   # 使用注意力参数对特征进行投射
        edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge，对应原论文中的添加leakyrelu操作
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        e_rowsum = spmm(edge, edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).cuda(device))  # e_rowsum: N x 1，spmm是稀疏矩阵和非稀疏矩阵的乘法操作
        h_prime = spmm(edge, edge_e, n=N,m=N, matrix=h)   # 把注意力评分与每个节点对应的特征相乘
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda(device))  # h_prime: N x out，div一看就是除，并且每一行的和要加一个9e-15防止除数为0
        # softmax结束
        if self.concat:
            # if this layer is not last layer
            return F.elu(h_prime)
        else:
            # if this layer is last layer
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.3, nheads=8):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        # self.residual = nn.Linear(nfeat,nclass,bias=False)
        self.attentions = [SparseGATLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SparseGATLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        # residual = self.residual(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



'''MLP'''
class MLP_onehot(nn.Module):
    def __init__(self,nfeat,nhid1,nhid2,nhid3,output,mlpdropout,mlpbn):
    # def __init__(self,nfeat,output):
        super(MLP_onehot, self).__init__()

        self.fc1 = nn.Linear(nfeat,output,bias=True)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(output,momentum=mlpbn)
        self.drop1 = nn.Dropout(mlpdropout)
        # self.fc2=nn.Linear(nhid1,nhid3)
        # self.norm2 = nn.BatchNorm1d(nhid3, momentum=mlpbn)
        # self.drop2 = nn.Dropout(mlpdropout)
        # self.fc3 = nn.Linear(nhid3,output)
        # self.norm3 = nn.BatchNorm1d(nhid3, momentum=mlpbn)
        # self.drop3 = nn.Dropout(mlpdropout)
        # self.fc4 = nn.Linear(nhid3, output,bias=False)
        # self.sig = nn.Sigmoid()
        # self.norm2 = nn.BatchNorm1d(int(nhid/4),momentum=mlpbn)
        # self.drop2 = nn.Dropout(mlpdropout)
        # self.fc3 = nn.Linear(int(nhid/4), output)
        # self.norm3 = nn.BatchNorm1d(int(nhid/5), momentum=mlpbn)
        # self.drop3 = nn.Dropout(mlpdropout)
        # self.fc4 = nn.Linear(int(nhid/5), output)
        # self.norm3 = nn.BatchNorm1d(nhid/8, momentum=mlpbn)
        # self.drop3 = nn.Dropout(mlpdropout)
        # self.fc4 = nn.Linear(nhid/8, output)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.norm1(x)
        # x = F.relu(x)
        # x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.norm2(x)
        # x = F.relu(x)
        # x = self.drop2(x)
        # x = self.fc3(x)
        # x = self.norm3(x)
        # x = F.relu(x)
        # x = self.drop3(x)
        # x = self.fc4(x)
        # x = self.sig(x)
        # x = self.fc1(x)

        # x=self.drop1(x)


        # x=F.relu(self.norm1(self.fc1(x)))


        # x=self.drop2(x)
        # x=self.norm2(x)
        # x=F.relu(x)
        # x=self.drop2(x)
        # x=self.fc3(x)
        # x=self.norm3(x)
        # x=F.relu(x)
        # x=self.drop3(x)
        # x=self.fc4(x)
        # x=F.log_softmax(x,dim=1)

        return x

class MLP(nn.Module):
    def __init__(self,nfeat,nhid,output,mlpdropout,mlpbn):
    # def __init__(self,nfeat,output):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(nfeat,nhid)
        self.norm1 = nn.BatchNorm1d(nhid,momentum=mlpbn)
        self.drop1 = nn.Dropout(mlpdropout)
        self.fc2=nn.Linear(nhid,int(nhid/3))

        self.norm2 = nn.BatchNorm1d(int(nhid/3), momentum=mlpbn)
        self.drop2 = nn.Dropout(mlpdropout)

        self.fc3 = nn.Linear(int(nhid/3), output)

    # self.sig = nn.Sigmoid()

        # self.norm2 = nn.BatchNorm1d(int(nhid/4),momentum=mlpbn)
        # self.drop2 = nn.Dropout(mlpdropout)
        # self.fc3 = nn.Linear(int(nhid/4), output)
        # self.norm3 = nn.BatchNorm1d(int(nhid/5), momentum=mlpbn)
        # self.drop3 = nn.Dropout(mlpdropout)
        # self.fc4 = nn.Linear(int(nhid/5), output)
        # self.norm3 = nn.BatchNorm1d(nhid/8, momentum=mlpbn)
        # self.drop3 = nn.Dropout(mlpdropout)
        # self.fc4 = nn.Linear(nhid/8, output)

    def forward(self,x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        # x = self.sig(x)
        # x = self.fc1(x)

        # x=self.drop1(x)


        # x=F.relu(self.norm1(self.fc1(x)))


        # x=self.drop2(x)
        # x=self.norm2(x)
        # x=F.relu(x)
        # x=self.drop2(x)
        # x=self.fc3(x)
        # x=self.norm3(x)
        # x=F.relu(x)
        # x=self.drop3(x)
        # x=self.fc4(x)
        # x=F.log_softmax(x,dim=1)

        return x

class AFStroke(BaseModel):
    def __init__(self,node_num,type_num,adj,emb_dim=16,gcn_layersize=[16,16,16],
                 dropout=0.5,mlpdropout=0.4,num_of_direction=2,num_of_layers=1,mlpbn=0.9,alpha=0.4,number_of_heads=8,gcn=False,drugbn=0.9,diseasebn=0.9):
        super().__init__()
        self.node_num = node_num
        self.adj = adj
        # self.graph = graph
        self.emb_dim = emb_dim
        self.alpha = alpha
        self.number_of_heads = number_of_heads
        self.num_of_direction=num_of_direction
        self.num_of_layers=num_of_layers
        self.value_embedding = nn.Embedding(node_num+1,emb_dim,padding_idx=0)
        self.type_embedding = nn.Embedding(type_num+1,emb_dim,padding_idx=0)
        # self.gcn = GCN(nfeat=gcn_layersize[0],nhid=gcn_layersize[1],nclass=gcn_layersize[2],dropout=dropout)
        # self.gcn = GAT(g=graph,in_dim=gcn_layersize[0],hidden_dim=gcn_layersize[1],out_dim=gcn_layersize[2])
        if gcn:
            self.gcn = GCN(nfeat=gcn_layersize[0], nhid=gcn_layersize[1], nclass=gcn_layersize[2], dropout=dropout)
        else:
            self.gcn = SpGAT(nfeat=gcn_layersize[0]
                           ,nhid=gcn_layersize[1],nclass=gcn_layersize[2],dropout=dropout,alpha=self.alpha,nheads=self.number_of_heads)

        self.lstm = nn.LSTM(input_size=emb_dim*2,hidden_size=emb_dim,num_layers=self.num_of_layers,
                            bidirectional=(self.num_of_direction==2))
        # self.bilstm = nn.LSTM(input_size=emb_dim*2,hidden_size=emb_dim,
        #                       num_layers=1,bidirectional=(self.num_of_direction==2))
        # self.gru = nn.GRU(input_size=emb_dim*2,hidden_size=emb_dim)
        ''''''
        self.stroke_lstm = nn.LSTM(input_size=emb_dim*2,hidden_size=emb_dim,num_layers=self.num_of_layers,
                            bidirectional=(self.num_of_direction==2))
        # self.stroke_bilstm = nn.LSTM(input_size=emb_dim*2,hidden_size=emb_dim,
        #                              num_layers=1,bidirectional=True)

        # self.stroke_gru = nn.GRU(input_size=emb_dim*2,hidden_size=emb_dim)
        ''''''

        self.node_attention_linear=nn.Linear(in_features=emb_dim,out_features=1,bias=False)
        self.node_attention_softmax = nn.Softmax(dim=1)
        ''''''
        self.stroke_node_attention_linear = nn.Linear(in_features=emb_dim,out_features=1,bias=False)
        self.stroke_node_attention_softmax = nn.Softmax(dim=1)
        ''''''
        # self.drugbn = nn.BatchNorm1d(emb_dim,momentum=drugbn)
        self.drug_path_attention_linear = nn.Linear(in_features=emb_dim,out_features=1,bias=False)
        self.drug_path_attention_softmax = nn.Softmax(dim=1)
        # self.disbn = nn.BatchNorm1d(emb_dim,momentum=diseasebn)
        self.disease_path_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.disease_attention_softmax = nn.Softmax(dim=1)
        # self.output_linear = nn.Linear(in_features=emb_dim*2,out_features=1)
        self.mlp=MLP((emb_dim*2),int(emb_dim*2/3),1,mlpdropout=mlpdropout,mlpbn=mlpbn)
        self.mlp_onehot = MLP_onehot(664+5,int(664/3),int(664/9),int(664/27),1,mlpdropout,mlpbn)
        self.wide_deep_w = nn.Linear(in_features=2,out_features=1,bias=True)
        # self.deep_Sig = nn.Sigmoid()
        # self.sigmoid = nn.Sigmoid()

    # output, _, _, _ = self.model(path_feature, type_feature, lengths, mask, stroke_feature, stroke_type_feature,
    #                              stroke_lengths_feature, stroke_mask_feature, gcn)

    def forward(self, path_feature,type_feature,lengths,mask,stroke_feature, stroke_type_feature
                ,stroke_lengths_feature, stroke_mask_feature,onehot_tensor,Sex_age_tensor,gcn=True):
        # shape pf path_feature:[batch_size,pathnum,path_length]
        #shape pf type_feature;[batch_size,path_num,path_length]
        '''GCN embedding'''
        total_node = torch.LongTensor([list(range(self.node_num+1))]).to(path_feature.device)
        ego_value_embedding = self.value_embedding(total_node).squeeze()
        if gcn:
            gcn_value_embedding = self.gcn(x=ego_value_embedding,adj=self.adj.to(path_feature.device))
            # gcn_value_embedding = self.gcn(x=ego_value_embedding,adj=self.adj.to(path_feature.device))
            # gcn_value_embedding = self.gcn(h = ego_value_embedding)
        else:
            gcn_value_embedding = ego_value_embedding

        '''embedding'''
        batch,path_num,path_len = path_feature.size()
        path_feature = path_feature.view(batch*path_num,path_len)
        #shape of path_embedding:[batch_size*path_num,path_length,emb_dim]
        path_embedding = gcn_value_embedding[path_feature]
        type_feature = type_feature.view(batch*path_num,path_len)
        #shape of type_embedding same with path
        type_embedding = self.type_embedding(type_feature).squeeze()
        #shape of feature ;[batch_size*path_num,path_length,emb_dim]
        feature = torch.cat((path_embedding,type_embedding),2)

        '''seperate stroke and drug path'''
        batch_stroke, path_stroke_num, path_stroke_len = stroke_feature.size()
        stroke_feature=stroke_feature.view(batch_stroke*path_stroke_num,path_stroke_len)
        stroke_path_embedding = gcn_value_embedding[stroke_feature]
        stroke_type_feature=stroke_type_feature.view(batch_stroke*path_stroke_num,path_stroke_len)
        stroke_type_embedding = self.type_embedding(stroke_type_feature).squeeze()
        feature_stroke = torch.cat((stroke_path_embedding,stroke_type_embedding),2)
        ''''''

        '''pack padded sequence'''
        feature = torch.transpose(feature,dim0=0,dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature,lengths=list(lengths.view(batch*path_num).data),
                                                 enforce_sorted=False)
        '''seperate lstm'''
        feature_stroke = torch.transpose(feature_stroke,dim0=0,dim1=1)
        feature_stroke = utils.rnn.pack_padded_sequence(feature_stroke,lengths=list(stroke_lengths_feature.view(batch_stroke*path_stroke_num).data),
                                                        enforce_sorted=False)
        ''''''
        # print(path_feature.device)
        h0 = torch.randn(self.num_of_layers * self.num_of_direction, batch*path_num, self.emb_dim).to(path_feature.device)
        c0 = torch.randn(self.num_of_layers * self.num_of_direction, batch*path_num, self.emb_dim).to(path_feature.device)


        '''LSTM'''
        lstm_out, (_,_) = self.lstm(feature,(h0,c0))
        # lstm_out, (_,_) = self.lstm(feature,(h0,c0))

        # lstm_out, _ = self.bilstm(feature)
        '''GRU'''
        # lstm_out, _ = self.gru(feature)
        ''''''
        lstm_out, _ = utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=path_len)
        if self.num_of_direction==2:
            (forward_out, backward_out) = torch.chunk(lstm_out, 2, dim=2)
            lstm_out = torch.concat((forward_out, backward_out), 1)

        '''LSTM Stroke'''
        # lstm_out_stroke, _ = self.stroke_gru(feature_stroke)
        # hs0 = torch.randn(self.num_of_layers * self.num_of_direction, batch*path_num, self.emb_dim).to(path_feature.device)
        # cs0 = torch.randn(self.num_of_layers * self.num_of_direction, batch*path_num, self.emb_dim).to(path_feature.device)
        h1 = torch.randn(self.num_of_layers * self.num_of_direction, batch_stroke * path_stroke_num, self.emb_dim).to(
            path_feature.device)
        c1 = torch.randn(self.num_of_layers * self.num_of_direction, batch_stroke * path_stroke_num, self.emb_dim).to(
            path_feature.device)
        lstm_out_stroke, (_,_) = self.stroke_lstm(feature_stroke,(h1,c1))

        lstm_out_stroke,_=utils.rnn.pad_packed_sequence(lstm_out_stroke,batch_first=True, total_length=path_stroke_len)
        if self.num_of_direction==2:
            (forward_out_stroke, backward_out_stroke) = torch.chunk(lstm_out_stroke, 2, dim=2)
            lstm_out_stroke = torch.concat((forward_out_stroke, backward_out_stroke), 1)





        '''node attention'''

        mask = mask.view(batch*path_num,path_len)
        if self.num_of_direction==2:
            mask = torch.concat([mask, mask], dim=1)

        output_path_embedding, drug_node_weight_normalized = self.drug_node_attention(input=lstm_out,mask=mask)
        drug_node_weight_normalized = drug_node_weight_normalized.view(batch,path_num,path_len*self.num_of_direction)
        output_path_embedding = output_path_embedding.view(batch,path_num,self.emb_dim)

        '''stroke node attention'''
        stroke_mask_feature = stroke_mask_feature.view(batch_stroke*path_stroke_num,path_stroke_len)
        if self.num_of_direction == 2:
            stroke_mask_feature = torch.concat([stroke_mask_feature, stroke_mask_feature], dim=1)

        output_stroke_path_embedding, stroke_node_weight_normalized = self.stroke_node_attention(input=lstm_out_stroke,mask=stroke_mask_feature)
        stroke_node_weight_normalized = stroke_node_weight_normalized.view(batch_stroke,path_stroke_num,path_stroke_len*self.num_of_direction)
        output_stroke_path_embedding = output_stroke_path_embedding.view(batch_stroke,path_stroke_num,self.emb_dim)


        '''path attention'''

        # output_path_embedding, path_weight_normalized = self.path_attention(output_path_embedding)
        # output_path_embedding_drug, path_weight_normalized = self.drug_path_attention(output_path_embedding,path_location)
        # output_path_embedding_disease, disease_weight_normalized = self.disease_path_attention(output_path_embedding,
        #                                                                                   disease_location)

        output_path_embedding_drug, drug_path_weight_normalized = self.drug_path_attention(output_path_embedding)
        output_stroke_path_embedding, disease_weight_normalized = self.disease_path_attention(output_stroke_path_embedding)
        # alpha = 0.5

        # output_embedding_test = output_path_embedding
        # output_embedding = output_path_embedding_disease

        # output_embedding = torch.concat([output_path_embedding_drug,output_path_embedding_disease],dim=1)
        # output_embedding = torch.concat([output_path_embedding_drug,output_stroke_path_embedding],dim=1)
        output_embedding = torch.concat([output_path_embedding_drug,output_stroke_path_embedding],dim=1)


        '''prediction'''
        output = self.mlp(output_embedding)
        # output_final = output
        onehot_tensor = torch.concat([onehot_tensor,Sex_age_tensor],dim=1)
        output2 = self.mlp_onehot(onehot_tensor)

        # output = self.deep_Sig(output)
        # output2 = self.deep_Sig(output2)


        output_final = torch.concat([output,output2],dim=1)
        output_final = self.wide_deep_w(output_final)

        # output_final = output

        # output_final_weight = self.wide_deep_w.weight.data
        # output_final_bias = self.wide_deep_w.bias.data
        # output_final = output

        # output_final = nn.Sigmoid(output+output2)
        # output = self.output_linear(output_embedding)
        # output = self.mlp(output_embedding)
        # output = self.sigmoid(output)
        # threshold = torch.tensor([0.3])
        # threshold = threshold.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # output = (output>threshold).float()*1

        return output_final,drug_node_weight_normalized,drug_path_weight_normalized,stroke_node_weight_normalized,disease_weight_normalized\
            ,output_embedding, _,_
               # output_final_weight, output_final_bias

    def drug_node_attention(self,input,mask):

        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.node_attention_linear(input) # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze()
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask==0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.node_attention_softmax(weight)
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)

        return input_weighted,weight_normalized

    def stroke_node_attention(self,input,mask):

        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.stroke_node_attention_linear(input) # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze()
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask==0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.stroke_node_attention_softmax(weight)
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)

        return input_weighted,weight_normalized

    '''path attention concating'''
    def drug_path_attention(self,input):
        #can add path location input future
        weight = self.drug_path_attention_linear(input)
        weight = weight.squeeze()
        # weight = weight.masked_fill(path_location==0,torch.tensor(-1e9))
        weight_normalized = self.drug_path_attention_softmax(weight)
        weight_expand = torch.unsqueeze(weight_normalized,dim=2)
        input_weighted = (input*weight_expand).sum(dim=1)
        # input_expand = input*weight_expand
        # output_list = torch.chunk(input_expand,path_num,dim=1)
        # output_cat = torch.cat(output_list,dim=2)
        # #output dimension is (batchsize, pathnum*emb_dim)
        # output_cat = output_cat.squeeze()

        return input_weighted,weight_normalized
    def disease_path_attention(self,input):
        #can add disease location in future

        weight = self.disease_path_attention_linear(input)
        weight = weight.squeeze()
        # weight = weight.masked_fill(disease_location==0,torch.tensor(-1e9))
        weight_normalized = self.disease_attention_softmax(weight)
        weight_expand = torch.unsqueeze(weight_normalized,dim=2)
        input_weighted = (input*weight_expand).sum(dim=1)

        return input_weighted,weight_normalized
