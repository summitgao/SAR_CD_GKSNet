import torch
from torch import nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io
import os
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def cosine_similarity1(x, y, norm=True):
    #""" 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y) #"len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))


    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

class GraphConvolution(nn.Module):

    def __init__(self,in_features=256,out_features=256,bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input,adj=None,relu=False):
        support = torch.matmul(input, self.weight)

        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support

        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ResNeXtUnit(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None, stride=1, groups=32):
        super(ResNeXtUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, mid_features, 3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class ResNeXt(nn.Module):
    def __init__(self, class_num):
        super(ResNeXt, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )  # 32x32
        self.stage_1 = nn.Sequential(
            ResNeXtUnit(64, 128, mid_features=128),
            nn.ReLU(),
        )  # 32x32
        self.stage_2 = nn.Sequential(
            ResNeXtUnit(128, 256, stride=2),
            nn.ReLU(),
 
        )  # 16x16
        self.stage_3 = nn.Sequential(
            ResNeXtUnit(256, 256, stride=2),
            nn.ReLU(),

        )  # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(256, class_num),
            # nn.Softmax(dim=1)
        )
        self.graph_conv1 = GraphConvolution(128, 128)
        self.graph_conv2 = GraphConvolution(128, 128)
        self.graph_conv3 = GraphConvolution(128, 128)
        self.fc_graph = GraphConvolution(128*3, 128)

    def forward(self, source, target):

        fea_source = self.basic_conv(source)
        fea_source = self.stage_1(fea_source)
        n1, c1, h1, w1 = fea_source.size()
        graph_source = fea_source.view(n1, h1 * w1, c1)
        graph_source = graph_source.data.cpu().numpy()

        fea_target = self.basic_conv(target)
        fea_target = self.stage_1(fea_target)
        n2, c2, h2, w2 = fea_target.size()
        graph_target = fea_target.view(n2, h2 * w2, c2)
        graph_target = graph_target.data.cpu().numpy()

        a_source = Parameter(torch.FloatTensor(h1 * w1, h1 * w1))
        a_source = torch.nn.init.xavier_uniform_(a_source)
        a_source = a_source.cuda()
        a_target = Parameter(torch.FloatTensor(h2 * w2, h2 * w2))
        a_target = torch.nn.init.xavier_uniform_(a_target)
        a_target = a_target.cuda()

        graph_source = torch.from_numpy(graph_source)
        graph_source = graph_source.cuda()
        graph_source1 = self.graph_conv1.forward(
            graph_source, adj=a_source, relu=True)
        graph_source2 = self.graph_conv2.forward(
            graph_source1, adj=a_source, relu=True)
        graph_source3 = self.graph_conv3.forward(
            graph_source2, adj=a_source, relu=True)

        graph_target = torch.from_numpy(graph_target)
        graph_target = graph_target.cuda()
        graph_target1 = self.graph_conv1.forward(
            graph_target, adj=a_target, relu=True)

        weight = Parameter(torch.FloatTensor(c1, c1))
        weight = torch.nn.init.xavier_uniform_(weight)
        weight = weight.cuda()

        graph_source = graph_source1.data.cpu().numpy()
        graph_target = graph_target1.data.cpu().numpy()

        a_tr = []
        for i in range(n1):
            b1 = cosine_similarity(graph_source[i], graph_target[i])
            a_tr.extend(b1)

        a_tr = np.array(a_tr, dtype=np.float32)
        a_tr = a_tr.reshape(n1, h2 * w2, h1 * w1)
        a_tr = torch.from_numpy(a_tr)
        a_tr = a_tr.cuda()

        source2target1 = torch.matmul(graph_source1, weight)
        source2target1 = torch.matmul(a_tr, source2target1)

        graph_target11 = torch.cat(
            (graph_source1, graph_target1, source2target1), dim=-1)
        graph_target11 = self.fc_graph.forward(graph_target11, relu=True)
        graph_target1 = graph_target1+graph_target11

        graph_target2 = self.graph_conv2.forward(
            graph_target1, adj=a_target, relu=True)
        source2target2 = torch.matmul(graph_source2, weight)
        source2target2 = torch.matmul(a_tr, source2target2)
        graph_target22 = torch.cat(
            (graph_source2, graph_target2, source2target2), dim=-1)
        graph_target22 = self.fc_graph.forward(graph_target22, relu=True)
        graph_target2=graph_target2+graph_target22

        graph_target3 = self.graph_conv3.forward(
            graph_target2, adj=a_target, relu=True)
        source2target3 = torch.matmul(graph_source3, weight)
        source2target3 = torch.matmul(a_tr, source2target3)
        graph_target33 = torch.cat(
            (graph_source3, graph_target3, source2target3), dim=-1)
        graph_target33 = self.fc_graph.forward(graph_target33, relu=True)
        graph_target3=graph_target3+graph_target33

        graph_output = graph_target3

        graph_output = graph_output.reshape(n2, c2, h2, w2)
        graph_output = graph_output.cuda()

        fea = fea_target + graph_output

        fea = self.stage_2(fea)
        #print("fea1",fea.size())
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)

        return fea

    
if __name__=='__main__':
    x = torch.rand(8,3,32,32)
    print(x.dtype)
    net = ResNeXt(10)
    out = net(x)
    print(out.shape)
