import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F

# DEBUG SWITCH
DEBUG_MODEL = False

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # anylygous with MFB
        self.Linear_predict = nn.Linear(1000, 3000)

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        # print("GraphConvolution-forward:\n")
        # print("(0)input.shape={0}\n".format(input.shape))
        # print("(1)adj.shape={0}\n".format(adj.shape))
        # print("adj=", adj, adj.dtype)
        # print("input=", input, input.dtype)
        # print("self.weight=",self.weight, self.weight.dtype)
        support = torch.matmul(input, self.weight)
        # print("support=", support, support.dtype)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):

        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.adj_file = adj_file
        self.opt = option
        self.state = {}
        self.state['use_gpu'] = torch.cuda.is_available()
        self.is_usemfb = option.IS_USE_MFB
        # self.is_only_fc = option.IS_ONLY_FC
        self.pooling_stride = option.pooling_stride
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        # create correlation matrix A pipeline
        self.d_e = in_channel
        self.d_e_1 = self.opt.inter_channel
        # conv2d parameters: in_channel,out_channel,kernel_size
        # usage: input the batchsize*in_channel*Height*Width
        # self.A_branch_1 = nn.Sequential(
        #     nn.Conv1d(self.d_e, self.d_e_1, kernel_size=1),      # kernel size is 1
        #     # nn.Softmax(),
        #     nn.Sigmoid(),
        # )
        # self.A_branch_2 = nn.Sequential(
        #     nn.Conv1d(self.d_e, self.d_e_1, kernel_size=1),      # kernel size is 1
        #     # nn.Softmax(),
        #     nn.Sigmoid(),
        # )
        self.A_branch_1 = nn.Sequential(
            nn.Conv2d(self.d_e, self.d_e_1, kernel_size=1),  # kernel size is 1
            # nn.Softmax(),
            nn.Sigmoid(),
        )
        self.A_branch_2 = nn.Sequential(
            nn.Conv2d(self.d_e, self.d_e_1, kernel_size=1),  # kernel size is 1
            # nn.Softmax(),
            nn.Sigmoid(),
        )
        # self.A_Linear_1 = nn.Linear(self.d_e, self.d_e_1)
        # self.A_Linear_2 = nn.Linear(self.d_e, self.d_e_1)
        ######################################
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # 2 layers of GCN
        # self.gc1 = GraphConvolution(in_channel, 1024)     # original ML_GCN 1st GCN layer
        ## for A_GCN
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        ## origin ML_GCN
        # t = self.opt.threshold_tao
        # _adj = gen_A(self.opt.threshold_p, num_classes, t, adj_file)
        ## my modified based ML_GCN
        # _adj = gen_A(self.opt.threshold_p, num_classes, self.opt.threshold_tao, adj_file)
        ### for A_GCN
        _adj = adj2tensor(adj_file)
        
        self.A = Parameter(torch.from_numpy(_adj).float())  # Parameter() is a function of torch.nn

        # image normalization
        # in the torchvision.transforms.Normalize  , it uses img=(img-mean)/std to get the every pixel value
        self.image_normalization_mean = [0.485, 0.456, 0.406]       # def mean
        self.image_normalization_std = [0.229, 0.224, 0.225]        # def std

        # Linear fusion structure
        # the imgae data projection
        # self.JOINT_EMB_SIZE = 8000
        self.JOINT_EMB_SIZE=option.linear_intermediate

        if self.is_usemfb:
            assert self.JOINT_EMB_SIZE % self.pooling_stride == 0, \
                'linear-intermediate value must can be divided exactly by sum pooling stride value!'
            self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
            # increase one fc layer for ML_task
            self.ML_fc_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), int(self.num_classes))
        else:
            self.out_in_tmp = int(1)
        
        self.Linear_imgdataproj = nn.Linear(option.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)  # IN=2048 ,OUT=JOINT_EMB_SIZE
        # the classifier data projection
        self.Linear_classifierproj = nn.Linear(option.CLASSIFIER_CHANNEL, self.JOINT_EMB_SIZE)  # IN=2048, OUT=JOINT_EMB_SIZE
        
        ### only for comparison experiment, only use one fc-layer instead of the GCN road and MFB module######
        # if self.is_only_fc:
        #     self.only_fc_layer = nn.Linear(option.IMAGE_CHANNEL, self.num_classes)
        ######################################################################################################

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)                 # the feature of every image is D = 2048
        ##################################################################################
        # if self.is_only_fc:
        #     return self.only_fc_layer(feature), 0.0
        ##################################################################################

        # Method of creating adj matrix in origin ML_GCN
        inp = inp[0]              # why here is `inp[0]`? inp[0] is same as inp[1]...
        # adj = gen_adj(self.A).detach()        # original adj matrix
        
        # correlation matrix A construction
        # print("in model.py forward func: the {0}.shape is :{1}".format("inp",inp.shape))    # inp.shape=[2,20,300]
        # print('inp is :\n{0}\n'.format(inp))
        # d_e*C , C denotes the num_classes
        # input_inp = inp.view((1,self.d_e, self.num_classes))     # input_inp as shape 1 × d_e × C× 1
        
        ## conv1d
        # branch_1 = self.A_branch_1(inp.view((1,self.d_e, self.num_classes))).view((self.d_e_1, self.num_classes))
        # branch_2 = self.A_branch_2(inp.view((1,self.d_e, self.num_classes))).view((self.d_e_1, self.num_classes))
        # A = torch.matmul(branch_1.t(), branch_2)/float(self.num_classes)

        ## conv2d
        branch_1 = self.A_branch_1(inp.view((1,self.d_e, self.num_classes, 1))).view((self.d_e_1, self.num_classes))
        branch_2 = self.A_branch_2(inp.view((1,self.d_e, self.num_classes, 1))).view((self.d_e_1, self.num_classes))
        A = torch.matmul(branch_1.t(), branch_2)/float(self.num_classes)
        
        # branch_1 = self.A_Linear_1(inp)
        # branch_2 = self.A_Linear_2(inp)
        # A = torch.matmul(branch_1, branch_2.t())/float(self.num_classes)
        # gen_correlation function
        I_c = torch.eye(A.shape[0]).cuda() if torch.cuda.is_available() else torch.eye(A.shape[0]).cpu()
        A_wave = A + I_c
        D_wave_negative_power = torch.diag(torch.pow(A_wave.sum(1).float(),-0.5))
        D_wave_negative_power[torch.isnan(D_wave_negative_power)] = 0.0
        D_wave_negative_power[torch.isinf(D_wave_negative_power)] = 0.0
        A_hat = torch.matmul(torch.matmul(D_wave_negative_power, A_wave), D_wave_negative_power)
        L_A_loss = torch.abs(A_hat - I_c).sum()
        if L_A_loss!=L_A_loss:
            print("A = \n", A)
            print("A_wave = \n", A_wave)
            print("A_wave sum(1) = \n", A_wave.sum(1))
            print("D~ diagnose elements = \n", torch.pow(A_wave.sum(1).float(),-0.5))
            print("D~ = \n", D_wave_negative_power)
            print("A_hat=\n", A_hat)
            sys.exit()
        # print("L_A_loss = ", L_A_loss)
        
        # adj = gen_correlation(A.cpu()).cuda() \
        #     if torch.cuda.is_available() else gen_correlation(A.cpu())
        adj = A_hat
        
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        # x shape is [20, 2048]

        x = x.transpose(0, 1)
        # x.shape = [2048, 20], feature.shape = [batchsize,2048]
        # fusion structure
        # x_out = torch.zeros(int(feature.shape[0]),int(self.num_classes)).cuda()
        if self.state['use_gpu']:
            x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
        else:
            x_out = torch.FloatTensor(torch.FloatStorage())
        if self.is_usemfb:
            for i_row in range(int(feature.shape[0])):
                # every row feature corresponding to one image
                # 1 * 2048 -> 1*linear_intermediate
                img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
                # print("(0)img_linear_row.shape = ", img_linear_row.shape)
                # out_row = torch.zeros(int(self.out_in_tmp * self.num_classes)).cuda()
                if self.state['use_gpu']:
                    out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
                else:
                    out_row = torch.FloatTensor(torch.FloatStorage())
                for col in range(int(x.shape[1])):  # 0 <= col <=19
                    # get one row of the x tensor
                    tmp_x = x[:, col].view(1, -1)  # 1 * 2048
                    classifier_linear = self.Linear_classifierproj(tmp_x)  # 1 * 2048 -> 1*linear_intermediate
                    # hadmard product
                    iq = torch.mul(img_linear_row, classifier_linear)  # 1*linear_intermediate, hadmard product
                    iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)  # drop out,ratio=0.1
                    iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2)  # 1 * 1 * linear_intermediate -> 1 * 1
                    # signed sqrt, this equal to iq ← sign(iq)(|iq|^0.5)
                    # iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))
                    # # iq = iq.view(-1,1)                      # 1 -> 1 * 1
                    # iq = F.normalize(iq)  # L-2 normalize
                    # out_row[col] = torch.squeeze(iq)
            
                    # catenate in terms of the column, shape (1,self.out_in_temp*self.num_classes)
                    out_row = torch.cat((out_row, iq), 1)
                # out_row is one row , dimension (1, num_classes*self.out_in_tmp)
                #
                if self.out_in_tmp != 1:  # if out_in_tmp == 1 , not use the ML_fc_layer
                    temp_out = self.ML_fc_layer(out_row)
                    out_row = temp_out  # out_row transformed to the shape (1,num_classes)
        
                # x_out[i_row, :] = out_rows
                x_out = torch.cat((x_out, out_row), 0)  # x_out is shape of (batchsize , num_classes)

        else:   x_out = torch.matmul(feature, x)      # batchsize * 20
        # print("x_out.shape={0},feature.shape={1}".format(x_out.shape, feature.shape))
        assert x_out.shape[0]==feature.shape[0]
        
        return x_out, L_A_loss

    def get_config_optim(self, lr, lrp):
        return [
                    ## Resnet101
                    {'params': self.features.parameters(), 'lr': lr * lrp},
                    ## generate correlation
                    {'params': self.A_branch_1.parameters(), 'lr': 4.0 * lr},
                    {'params': self.A_branch_2.parameters(), 'lr': 4.0 * lr},
                    # {'params': self.A_Linear_1.parameters(), 'lr': lr * lrp},
                    # {'params': self.A_Linear_2.parameters(), 'lr': lr * lrp},
                    ## MFB-ism
                    # {'params': self.Linear_imgdataproj.parameters(), 'lr': lr},
                    # {'params': self.Linear_classifierproj.parameters(), 'lr': lr},
                    # {'params': self.ML_fc_layer.parameters(), 'lr': lr},
                    ## GCN
                    {'params': self.gc1.parameters(), 'lr': lr},
                    {'params': self.gc2.parameters(), 'lr': lr},
                ]

def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    '''
    :param opt:         from config.py
    :param num_classes: the amount of num_classes
    :param t:           this value corresponding with the "threshold tao" when construct the correlation matrix
    :param pretrained:  use pretrained resnet101 or not
    :param adj_file:    /data/voc/voc_adj.pkl file or /data/coco/coco_adj.pkl file
    :param in_channel:  input dimensionality
    :return:
    '''
    model = models.resnet101(pretrained=pretrained)
    ### iff the computer cannot connect Internet, we must load pretraind model use the following tow rows codes
    # model = models.resnet101(pretrained=False)
    # model.load_state_dict(torch.load('./checkpoint/voc/voc_checkpoint.pth.tar'))
    ##################################################################

    return GCNResnet(opt, model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)