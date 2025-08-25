import os
import time
import cv2
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_wavelets import DWTInverse, DWTForward
from torchvision import transforms as T
normalize=T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])

######################################3

def Unit_Normalize(img,eps=0.0000000000000001):
    a=torch.zeros(1,1)
    b=np.zeros((1,1))
    if type(img)==type(b):
        return (((img - np.min(img)) / (np.max(img) - np.min(img)+eps)) * 255).astype(np.uint8)
    elif type(img)==type(a):
        if len(img.shape)==2 or len(img.shape)==3:
            return ((img - torch.min(img))  / (torch.max(img) - torch.min(img)+eps))* 255
        elif len(img.shape)==4:
            B,C,H,W=img.shape
            IMG=[]
            for i in range (B):
                IMG.append(((img[i] - torch.min(img[i]))  / (torch.max(img[i]) - torch.min(img[i])+eps))* 255)
            return torch.stack(IMG,dim=0)

###############################################################################


################Loss functions###################
class SWBCE(nn.Module):
    def __init__(self, Label_Pred_balance=1, balance=1.1, l_weight=[1.1]):
        # Pred can be an image or a list, Label_Pred_balance is the trade-off between CWBCE_Label (i.e. WBCE) and CWBCE_Pred, set to 0 if only use WBCE simply.
        super(SWBCE, self).__init__()
        self.Label_Pred_balance = Label_Pred_balance
        self.balance = balance
        self.l_weight = l_weight

    def forward(self, inputs, targets):
        if type(inputs)==type([]):
            cost=0
            for i in range(len(inputs)):
                cost = (self.CWBCE_Label(inputs[i], targets,l_weight=self.l_weight[i]).mean() + self.Label_Pred_balance * self.CWBCE_Pred(inputs[i], targets, l_weight=self.l_weight[i]).mean()) / (1 + self.Label_Pred_balance)
        else:
            cost = (self.CWBCE_Label(inputs,targets,l_weight=self.l_weight[-1]).mean()+self.Label_Pred_balance*self.CWBCE_Pred(inputs,targets,l_weight=self.l_weight[-1]).mean())/(1+self.Label_Pred_balance)
        return cost

    def CWBCE_Label(self,inputs, targets, l_weight=1.1, Mask=None):
        b, c, h, w = targets.shape
        mask = targets.float().clamp(0, 1)
        num_positive = mask.sum()
        num_negative = b * c * h * w - num_positive
        weight_positive = 1.0 * num_negative / (num_positive + num_negative)
        weight_negative = self.balance * num_positive / (num_positive + num_negative)
        weight = mask * weight_positive + (1 - mask) * weight_negative
        if Mask != None:
            weight = weight * Mask
        cost = torch.nn.BCELoss(weight, reduction='none')(inputs, targets.float())
        return l_weight * cost

    def CWBCE_Pred(self,inputs, targets, l_weight=1.1, Mask=None):
        b, c, h, w = targets.shape
        mask = inputs.float().clamp(0, 1)
        num_positive = mask.sum()
        num_negative = b * c * h * w - num_positive
        weight_positive = 1.0 * num_negative / (num_positive + num_negative)
        weight_negative = self.balance * num_positive / (num_positive + num_negative)
        weight = inputs * weight_positive + (1 - inputs) * weight_negative
        if Mask != None:
            weight = weight * Mask
        cost = weight * torch.nn.BCELoss(reduction='none')(inputs, targets.float())
        return l_weight * cost

#
# def WBCE(inputs, targets, l_weight=[1.1]):# Pred can be an image or a list
#     targets = targets.long()
#     mask = targets.float()
#     num_positive = torch.sum((mask > 0.5).float()).float()
#     num_negative = torch.sum((mask <= 0.5).float()).float()
#     mask[mask > 0.5] = 1.0 * num_negative / (num_positive + num_negative)
#     mask[mask <= 0.5] = 1.1 * num_positive / (num_positive + num_negative)
#     if type(inputs)==type([]):
#         cost=0
#         for i in range(len(inputs)):
#             cost = cost+l_weight[i] *torch.nn.BCELoss(mask, reduction='mean')(inputs[i], targets.float())
#     else:
#         cost = l_weight[-1] * torch.nn.BCELoss(mask, reduction='mean')(inputs, targets.float())
#     return cost

#############################SWBCE##########################
#
# def CWBCE_Label(inputs, targets, balance=1.1,l_weight=1.1,Mask=None):# Pred should be an image
#     b,c,h,w=targets.shape
#     mask = targets.float().clamp(0,1)
#     num_positive = mask.sum()
#     num_negative = b*c*h*w-num_positive
#     weight_positive = 1.0 * num_negative / (num_positive + num_negative)
#     weight_negative = balance * num_positive / (num_positive + num_negative)
#     weight=mask*weight_positive+(1-mask)*weight_negative
#     if Mask!=None:
#         weight=weight*Mask
#     cost = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets.float())
#     return l_weight * cost
#
# def CWBCE_Pred(inputs, targets, balance=1.1,l_weight=1.1,Mask=None):# Pred should be an image
#     b,c,h,w=targets.shape
#     mask = inputs.float().clamp(0,1)
#     num_positive = mask.sum()
#     num_negative = b*c*h*w-num_positive
#     weight_positive = 1.0 * num_negative / (num_positive + num_negative)
#     weight_negative = balance * num_positive / (num_positive + num_negative)
#     weight=inputs*weight_positive+(1-inputs)*weight_negative
#     if Mask!=None:
#         weight=weight*Mask
#     cost = (weight*torch.nn.BCELoss(reduction='none')(inputs, targets.float())).mean()
#     return l_weight * cost
#
# def SWBCE_1(inputs, targets,balance=1,Mask=None):# Pred should be an image
#     cost=(CWBCE_Label(inputs,targets,Mask=Mask)+balance*CWBCE_Pred(inputs,targets,Mask=Mask))/(1+balance)
#     return cost

class BAA_Loss(nn.Module):#Pred can be either an image or a list
    def __init__(self, thr=0.7,thr_dev=0.2,res=1,bend=16,base_loss='WBCE',Disable_multi_layer=True,Only_Last_Thr=True,balance=1.1,l_weight=[1.1],device='cpu',edge_r=7,EBT_balance=[1,0.8,0.5]):
                                                                          #Only use last layer     #When use multi-layer, only last Thr
        super(Thr_Loss, self).__init__()
        self.thr_dev = thr_dev
        self.bend = bend
        self.thr = thr
        self.Only_Last_Thr=Only_Last_Thr#多输出时只在最后一层Thr
        self.Disable_multi_layer=Disable_multi_layer#Thr但不启用多层优化
        assert self.thr_dev >= 0 and self.bend > 0 and self.thr>=0
        self.balance = balance
        self.l_weight = l_weight
        self.Fun_Coe=math.exp(self.bend*self.thr_dev)
        self.res = res
        self.base_loss=base_loss
        assert self.base_loss=='WBCE' or self.base_loss=='SWBCE' or self.base_loss=='EBT'
        if self.base_loss=='EBT':
            self.device=device
            self.edge_r = edge_r
            self.EBT_balance = EBT_balance  # E,B,T系数
            self.filt = torch.ones(1, 1, 2 * self.edge_r + 1, 2 * self.edge_r + 1)
            self.filt.requires_grad = False
            self.filt = self.filt.to(self.device)

    def forward(self, Pred, label):
        if type(Pred)==type([]):
            loss=0
            if self.Disable_multi_layer==True:
                Mask = self.Der_Mask(inputs=Pred[-1], targets=label) + self.res
                if self.base_loss == 'WBCE':
                    loss = loss+self.CWBCE_Label(inputs=Pred[-1], targets=label, l_weight=self.l_weight[-1], Mask=Mask)
                elif self.base_loss == 'SWBCE':
                    loss = loss+self.SyCWBCE_1(inputs=Pred[-1], targets=label, l_weight=self.l_weight[-1], Mask=Mask)
                elif self.base_loss == 'EBT':
                    loss = loss+self.EBT_Loss(Pred=Pred[-1], label=label, l_weight=self.l_weight[-1],Mask=Mask)
            elif self.Disable_multi_layer==False and self.Only_Last_Thr==False:
                for i in range(len(Pred)):
                    Mask = self.Der_Mask(inputs=Pred[i], targets=label) + self.res
                    if self.base_loss == 'WBCE':
                        loss = loss+self.CWBCE_Label(inputs=Pred[i], targets=label, l_weight=self.l_weight[i], Mask=Mask)
                    elif self.base_loss == 'SWBCE':
                        loss = loss+self.SyCWBCE_1(inputs=Pred[i], targets=label, l_weight=self.l_weight[i], Mask=Mask)
                    elif self.base_loss == 'EBT':
                        loss = loss+self.EBT_Loss(Pred=Pred[i], label=label, l_weight=self.l_weight[i],Mask=Mask)
            elif self.Disable_multi_layer==False and self.Only_Last_Thr==True:
                for i in range(len(Pred)):
                    if i==len(Pred)-1:
                        Mask = self.Der_Mask(inputs=Pred[i], targets=label) + self.res
                    else:
                        Mask=None
                    if self.base_loss == 'WBCE':
                        loss = loss+self.CWBCE_Label(inputs=Pred[i], targets=label, l_weight=self.l_weight[i], Mask=Mask)
                    elif self.base_loss == 'SWBCE':
                        loss = loss+self.SyCWBCE_1(inputs=Pred[i], targets=label, l_weight=self.l_weight[i], Mask=Mask)
                    elif self.base_loss == 'EBT':
                        loss = loss+self.EBT_Loss(Pred=Pred[i], label=label, l_weight=self.l_weight[i],Mask=Mask)
        else:
            Mask = self.Der_Mask(inputs=Pred, targets=label)+self.res
            if self.base_loss=='WBCE':
                loss = self.CWBCE_Label(inputs=Pred,targets=label,l_weight=self.l_weight[0],Mask=Mask)
            elif self.base_loss=='SWBCE':
                loss = self.SyCWBCE_1(inputs=Pred, targets=label, l_weight=self.l_weight[0], Mask=Mask)
            elif self.base_loss=='EBT':
                loss = self.EBT_Loss(Pred=Pred, label=label, l_weight=self.l_weight[0], Mask=Mask)

        return loss

    def Der_Mask(self, inputs, targets):
        mask = targets.float().clamp(0, 1)
        distance = (inputs - self.thr) * mask + (self.thr - inputs) * (1 - mask)
        Weight = self.Distance_Weight_Function(distance=distance)
        return Weight

    def Distance_Weight_Function(self,distance):#distance 为到实际值到边界的距离,tensor，同侧>0，另侧<0， endvalue 为边界宽度（大于 endvalue 认为很好）， bend>0 为弯曲程度
        out=(torch.exp(self.bend*distance)-self.Fun_Coe)/(1-self.Fun_Coe)
        return out.clamp(0,1)

    def CWBCE_Label(self,inputs, targets,l_weight,Mask=None):
        b,c,h,w=targets.shape
        mask = targets.float().clamp(0,1)
        num_positive = mask.sum()
        num_negative = b*c*h*w-num_positive
        weight_positive = 1.0 * num_negative / (num_positive + num_negative)
        weight_negative = self.balance * num_positive / (num_positive + num_negative)
        weight=mask*weight_positive+(1-mask)*weight_negative
        if Mask!=None:
            weight=weight*Mask
        cost = (weight * torch.nn.BCELoss(reduction='none')(inputs, targets.float())).mean()
        return l_weight * cost

    def CWBCE_Pred(self,inputs, targets,l_weight,Mask=None):
        b,c,h,w=targets.shape
        mask = inputs.float().clamp(0,1)
        num_positive = mask.sum()
        num_negative = b*c*h*w-num_positive
        weight_positive = 1.0 * num_negative / (num_positive + num_negative)
        weight_negative = self.balance * num_positive / (num_positive + num_negative)
        weight=inputs*weight_positive+(1-inputs)*weight_negative
        if Mask!=None:
            weight=weight*Mask
        cost = (weight*torch.nn.BCELoss(reduction='none')(inputs, targets.float())).mean()
        return l_weight * cost

    def SyCWBCE_1(self,inputs, targets,balance=1,l_weight=1.1,Mask=None):
        cost=(self.CWBCE_Label(inputs,targets,l_weight=l_weight,Mask=Mask)+balance*self.CWBCE_Pred(inputs,targets,l_weight=l_weight,Mask=Mask))/(1+balance)
        return cost

    def EBT_Loss(self, Pred, label,l_weight=1,Mask=None):

        b, c, h, w = label.shape
        num_total = b * c * h * w

        mask = label.float().clamp(0, 1)
        Edge = F.conv2d(mask, self.filt, bias=None, stride=1, padding=self.edge_r)  # 取边界及附近
        Edge[Edge > 0] = 1.0
        B = Edge - label  # 取边界附近（无边界）
        Texture = 1 - Edge  # 远离边界

        num_E = mask.sum()
        num_B = B.sum()
        num_T = Texture.sum()

        weight_E = (1 - (num_E / num_total)) * self.EBT_balance[0] * mask
        weight_B = (1 - (num_B / num_total)) * self.EBT_balance[1] * B
        weight_T = (1 - (num_T / num_total)) * self.EBT_balance[2] * Texture

        weight = weight_E + weight_B + weight_T
        if Mask!=None:
            weight=weight*Mask

        loss = l_weight*self.WBCE(inputs=Pred, targets=label, weight=weight).mean()
        return loss

    def WBCE(self, inputs, targets, weight=1):
        loss = weight * torch.nn.BCELoss(reduction='none')(inputs, targets.float())
        return loss

class EBT_WBCE(nn.Module):# The EBT loss, Pred can be an image or a list
    def __init__(self,device='cuda',edge_r=7,EBT_balance=[1,0.8,0.5],layer_weight=[1]):
        super(EBT_WBCE, self).__init__()
        self.device=device
        self.edge_r = edge_r
        self.EBT_balance=EBT_balance
        self.layer_weight =layer_weight

        self.filt = torch.ones(1, 1, 2 * self.edge_r + 1, 2 * self.edge_r + 1)
        self.filt.requires_grad = False
        self.filt = self.filt.to(self.device)

    def forward(self, Pred, label):
        if type(Pred)==type([]):
            loss=0
            for i in range(len(Pred)):
                loss=loss+self.layer_weight[i]*self.EBT_Loss(Pred[i],label)
        else:
            loss = self.layer_weight[0] * self.EBT_Loss(Pred, label)

        return loss

    def EBT_Loss(self, Pred, label):

        b, c, h, w = label.shape
        num_total=b*c*h*w

        mask = label.float().clamp(0, 1)
        Edge = F.conv2d(mask, self.filt, bias=None, stride=1, padding=self.edge_r)
        Edge[Edge > 0] = 1.0
        B = Edge - label
        Texture = 1 - Edge

        num_E=mask.sum()
        num_B=B.sum()
        num_T=Texture.sum()

        weight_E=(1-(num_E/num_total))*self.EBT_balance[0]*mask
        weight_B=(1-(num_B/num_total))*self.EBT_balance[1]*B
        weight_T=(1-(num_T/num_total))*self.EBT_balance[2]*Texture

        weight=weight_E+weight_B+weight_T

        loss=self.WBCE(inputs=Pred,targets=label,weight=weight).mean()
        return loss

    def WBCE(self,inputs, targets,weight=1):
        loss=weight * torch.nn.BCELoss(reduction='none')(inputs, targets.float())
        return loss

class Tracing_loss(nn.Module):#for HED, BDCN, RCF, etc, Pred can be a single image or a list
    def __init__(self,device='cuda',balance=1.1,tex_factor=[0.02], bdr_factor=[4]):
        super(Tracing_loss, self).__init__()
        self.device=device
        self.mask_radius = 2
        self.radius = 2
        self.balance=balance

        self.filt = torch.ones(1, 1, 2 * self.radius + 1, 2 * self.radius + 1)
        self.filt.requires_grad = False
        self.filt = self.filt.to(self.device)

        self.filt1 = torch.ones(1, 1, 3, 3)
        self.filt1.requires_grad = False
        self.filt1 = self.filt1.to(self.device)
        self.filt2 = torch.ones(1, 1, 2 * self.mask_radius + 1, 2 * self.mask_radius + 1)
        self.filt2.requires_grad = False
        self.filt2 = self.filt2.to(self.device)

        self.tex_factor = tex_factor
        self.bdr_factor=bdr_factor

    def forward(self, Pred, label):
        if type(Pred)==type([]):
            loss=0
            for i in range(len(Pred)):
                loss=loss+self.tracingloss(Pred[i], label, tex_factor=self.tex_factor[i], bdr_factor=self.bdr_factor[i])
        else:
            loss =self.tracingloss(Pred, label, tex_factor=self.tex_factor[-1], bdr_factor=self.bdr_factor[-1])
        return loss

    def bdrloss(self,prediction, label):
        bdr_pred = prediction * label
        pred_bdr_sum = label * F.conv2d(bdr_pred, self.filt, bias=None, stride=1, padding=self.radius)
        texture_mask = F.conv2d(label.float(), self.filt, bias=None, stride=1, padding=self.radius)
        mask = (texture_mask != 0).float()
        mask[label == 1] = 0
        pred_texture_sum = F.conv2d(prediction * (1-label) * mask, self.filt, bias=None, stride=1, padding=self.radius)
        softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
        cost = -label * torch.log(softmax_map)
        cost[label == 0] = 0
        return cost.sum()

    def textureloss(self,prediction, label):
        pred_sums = F.conv2d(prediction.float(), self.filt1, bias=None, stride=1, padding=1)
        label_sums = F.conv2d(label.float(), self.filt2, bias=None, stride=1, padding=self.mask_radius)
        mask = 1 - torch.gt(label_sums, 0).float()
        loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
        loss[mask == 0] = 0
        return torch.sum(loss)

    def tracingloss(self,prediction, label, tex_factor=0., bdr_factor=0.):
        label = label.float()
        prediction = prediction.float()
        with torch.no_grad():
            mask = label.clone()
            num_positive = torch.sum((mask==1).float()).float()
            num_negative = torch.sum((mask==0).float()).float()
            beta = num_negative / (num_positive + num_negative)
            mask[mask == 1] = beta
            mask[mask == 0] = self.balance * (1 - beta)
            mask[mask == 2] = 0
        cost = torch.sum(torch.nn.functional.binary_cross_entropy(prediction.float(),label.float(), weight=mask, reduce=False))
        label_w = (label != 0).float()
        textcost = self.textureloss(prediction.float(),label_w.float())
        bdrcost = self.bdrloss(prediction.float(),label_w.float())
        return cost + bdr_factor*bdrcost + tex_factor*textcost

class RankLoss_Last(nn.Module):

    def __init__(self, device='cpu'):#Only use the last layer for save GPU memory
        super(RankLoss_Last, self).__init__()
        self.device = device

    def forward(self, Pred, targets, delta=0.1, split=4):#logits=pred
        if type(Pred)==type([]):
            logits=Pred[-1]
        else:
            logits = Pred
        B, C, W, H = logits.size()
        logits = logits.view(B, -1)
        targets = targets.view(B, -1)
        # loss_weight = torch.exp(1-targets)
        # Filter fg logits
        fg_labels = (targets > 0)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)
        # fg_targets = targets[fg_labels]

        if fg_num != 0:
            threshold_logit = torch.min(fg_logits) - delta
            relevant_bg_labels = ((torch.logical_not(fg_labels)) & (logits >= threshold_logit))
            relevant_bg_logits = logits[relevant_bg_labels]

            ranking_error = torch.zeros(fg_num).to(self.device)
            fg_logits_sorted, sorted_indices = torch.sort(fg_logits)

            start = 0
            end = fg_num // split
            for ii in range(split):
                fg_relations = fg_logits - fg_logits_sorted[start:end, None]#[fg_logits-fg_logits_sorted[start],fg_logits-fg_logits_sorted[start+1],...,fg_logits-fg_logits_sorted[end-1]]
                fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)

                bg_relations = relevant_bg_logits - fg_logits_sorted[start:end, None]
                bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)

                rank_pos = torch.sum(fg_relations, dim=1)
                FP_num = torch.sum(bg_relations, dim=1)

                rank = rank_pos + FP_num
                ranking_error[start:end] = FP_num / rank

                start = end
                if ii == split - 2:
                    end = fg_num
                else:
                    end *= 2
            return ranking_error.mean()
        else:
            return (logits-logits).mean()

#####################################################################################


###############################Model Blocks###################################
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Flatten_Transpose(nn.Module):

    def __init__(self):
        super(Flatten_Transpose,self).__init__()

    def forward(self,x):
        B,C,H,W = x.size()
        x=x.view(B,-1,W*H).permute(0,2,1)

        return x

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x

class Upsample(nn.Module):

    def __init__(self, in_channel, out_channel, bn=False, mode='bicubic'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
        self.act = nn.LeakyReLU()
        self.bn = bn
        if bn:
            self.bn1 = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv2(x)
        return x

class Downsample(nn.Module):

    def __init__(self, in_channel, out_channel, bn=False):
        super(Downsample, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1)
        self.act = nn.LeakyReLU()
        self.bn = bn
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x

##################Non-position_transformer###################

class Transformer_Encoder(nn.Module):
    def __init__(self,in_channel,num_heads=8,drop=0.0):
        super(Transformer_Encoder,self).__init__()
        self.FT=Flatten_Transpose()
        self.Att = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True, add_bias_kv=False,
                                    add_zero_attn=False,kdim=None, vdim=None, batch_first=True, device=None, dtype=None)
        self.layer_norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.MLP=PositionwiseFeedForward(d_in=in_channel,d_hid=in_channel*4,dropout=drop)

    def forward(self, x):
        B,C,H,W=x.shape
        x=self.FT(x)#[B,HW,C]
        out,_=self.Att(x,x,x,need_weights= False)
        out=self.layer_norm(out+x)
        out=self.MLP(out).permute(0,2,1).view(B,C,H,W)

        return out

class Transformer_Decoder(nn.Module):

    def __init__(self, in_channel,num_heads=8,drop=0.0):
        super(Transformer_Decoder, self).__init__()

        self.Att1 = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True,
                                         add_bias_kv=False,
                                         add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None,
                                         dtype=None)
        self.layer_norm1= nn.LayerNorm(in_channel, eps=1e-6)

        self.Att2 = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True,
                                          add_bias_kv=False,
                                          add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None,
                                          dtype=None)
        self.layer_norm2= nn.LayerNorm(in_channel, eps=1e-6)

        self.MLP = PositionwiseFeedForward(d_in=in_channel, d_hid=in_channel * 4, dropout=drop)

        self.FT=Flatten_Transpose()

    def forward(self, x, enc_output):
        B,C,H,W=x.shape

        x=self.FT(x)
        out,_= self.Att1(x,x,x,need_weights= False)
        x=self.layer_norm1(out+x)

        enc_output=self.FT(enc_output)
        out, _ = self.Att2(enc_output, enc_output, x, need_weights=False)
        x = self.layer_norm2(out + x)

        x = self.MLP(x).permute(0,2,1).view(B,C,H,W)

        return x

class Transformer_Encoder_Decoder(nn.Module):

    def __init__(self, in_channel,num_encoder=6,num_decoder=6,num_heads=8,drop=0.0):
        super(Transformer_Encoder_Decoder, self).__init__()

        self.num_encoder=num_encoder
        self.num_decoder=num_decoder

        self.Encorder_list = nn.ModuleList([])
        for i in range(num_encoder):
            self.Encorder_list.append(Transformer_Encoder(in_channel=in_channel,num_heads=num_heads,drop=drop))

        self.Decorder_list = nn.ModuleList([])
        for i in range(num_decoder):
            self.Decorder_list.append(Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop))

    def forward(self, x):
        y=x
        for i in range (self.num_encoder):
            y=self.Encorder_list[i](y)
        for i in range (self.num_decoder):
            x=self.Decorder_list[i](x,y)
        return x

######################################################


##############################Selector Moels#######################

class Selecting_Weight_ED_Num(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2):
        super(Selecting_Weight_ED_Num, self).__init__()
        self.mode=mode
        self.trans_level=trans_level

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        if trans_level>=5:
            self.trans1 = Transformer_Encoder_Decoder(in_channel=32, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)
        if trans_level>=4:
            self.trans2 = Transformer_Encoder_Decoder(in_channel=64, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)
        if trans_level>=3:
            self.trans3 = Transformer_Encoder_Decoder(in_channel=128, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        if trans_level>=2:
            self.trans4 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        if trans_level>=1:
            self.trans5 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)
        if self.trans_level>=5:
            x = self.trans1(x)

        x2 = self.down1(x)
        if self.trans_level>=4:
            x2 = self.trans2(x2)

        x3=self.down2(x2)
        if self.trans_level>=3:
            x3 = self.trans3(x3)

        x4=self.down3(x3)
        if self.trans_level>=2:
            x4 = self.trans4(x4)

        x5=self.down4(x4)
        if self.trans_level>=1:
            x5 = self.trans5(x5)

        y=(self.up5(x5)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Selecting_Weight_ED_Num_Double(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2):
        super(Selecting_Weight_ED_Num_Double, self).__init__()
        self.mode=mode
        self.trans_level=trans_level

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        if trans_level>=5:
            self.trans1 = Transformer_Encoder_Decoder(in_channel=64, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down1=Downsample(in_channel=64,out_channel=128,bn=bn)
        if trans_level>=4:
            self.trans2 = Transformer_Encoder_Decoder(in_channel=128, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down2=Downsample(in_channel=128,out_channel=256,bn=bn)
        if trans_level>=3:
            self.trans3 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down3=Downsample(in_channel=256,out_channel=512,bn=bn)
        if trans_level>=2:
            self.trans4 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=512,out_channel=1024,bn=bn)
        if trans_level>=1:
            self.trans5 = Transformer_Encoder_Decoder(in_channel=1024, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=1024,out_channel=512,bn=bn)
        self.up4 = Upsample(in_channel=512, out_channel=256,bn=bn)
        self.up3 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up2 = Upsample(in_channel=128, out_channel=64,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(64)
            self.bnlast_1 = nn.BatchNorm2d(128)
            self.bnlast_2 = nn.BatchNorm2d(256)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)
        if self.trans_level>=5:
            x = self.trans1(x)

        x2 = self.down1(x)
        if self.trans_level>=4:
            x2 = self.trans2(x2)

        x3=self.down2(x2)
        if self.trans_level>=3:
            x3 = self.trans3(x3)

        x4=self.down3(x3)
        if self.trans_level>=2:
            x4 = self.trans4(x4)

        x5=self.down4(x4)
        if self.trans_level>=1:
            x5 = self.trans5(x5)

        y=(self.up5(x5)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

#############################################################


##########################Belows to the end are for structure testing of the ES#######

###############Rot_transformer####################

def precompute_freqs_cis(c, h, w,theta=100000000.0):  # channel,h,w, img

    freqs = 1.0 / (theta ** (torch.arange(0, c, 4)[: (c // 4)].float() / c))

    Ones = torch.ones(h, w)
    t_h = torch.arange(h)
    T_h = (Ones.permute(1, 0) * t_h).permute(1, 0).unsqueeze(2)

    t_w = torch.arange(w)
    T_w = (Ones * t_w).unsqueeze(2)

    T = torch.cat([T_h, T_w], dim=2)

    T_final = T * freqs[0]
    for i in range(1, len(freqs)):
        T_final = torch.cat([T_final, T * freqs[i]],dim=2)
    freqs_cis = torch.polar(torch.ones_like(T_final),T_final)
    return freqs_cis

def apply_rotary_emb(
                     xq: torch.Tensor,
                     xk: torch.Tensor,
                     freqs_cis: torch.Tensor,  # q,k,f, flatten
                     ):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous()
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous()

    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)  # [B,N,D]

class Rot_MultiheadAttention(nn.Module):

    def __init__(self, hid_dim, n_heads,C,H,W, dropout=0.0, theta = 100000000.0):
        super(Rot_MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % (2*n_heads) == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = hid_dim // n_heads

        self.precompute=precompute_freqs_cis(C, H, W,theta=theta)

    def forward(self, query, key, value,h=None,w=None,c=None):

        precompute=self.precompute[0:h,0:w,:].reshape(h * w, c // 2)
        #
        # precompute=precompute_freqs_cis(c, h, w)
        # if precompute.device!=query.device:
        #     precompute=precompute.to(query.device)

        if precompute.device!=query.device:
            precompute=precompute.to(query.device)

        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q,K=apply_rotary_emb(Q,K,precompute)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = self.do(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        return x

class Rot_Transformer_Encoder(nn.Module):
    def __init__(self,in_channel,h,w,num_heads=8,drop=0.0):
        super(Rot_Transformer_Encoder,self).__init__()
        self.FT=Flatten_Transpose()
        self.Att = Rot_MultiheadAttention(hid_dim=in_channel,n_heads=num_heads,dropout=drop,C=in_channel,H=h,W=w)
        self.layer_norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.MLP=PositionwiseFeedForward(d_in=in_channel,d_hid=in_channel*4,dropout=drop)

    def forward(self, x):
        B,C,H,W=x.shape
        x=self.FT(x)#[B,HW,C]
        out=self.Att(x,x,x,c=C,h=H,w=W)
        out=self.layer_norm(out+x)
        out=self.MLP(out).permute(0,2,1).view(B,C,H,W)

        return out

class Rot_Transformer_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_heads=8,drop=0.0):
        super(Rot_Transformer_Decoder, self).__init__()

        self.Att1 = Rot_MultiheadAttention(hid_dim=in_channel, n_heads=num_heads, dropout=drop,C=in_channel,H=h,W=w)
        self.layer_norm1= nn.LayerNorm(in_channel, eps=1e-6)

        self.Att2 = Rot_MultiheadAttention(hid_dim=in_channel, n_heads=num_heads, dropout=drop,C=in_channel,H=h,W=w)
        self.layer_norm2= nn.LayerNorm(in_channel, eps=1e-6)

        self.MLP = PositionwiseFeedForward(d_in=in_channel, d_hid=in_channel * 4, dropout=drop)

        self.FT=Flatten_Transpose()

    def forward(self, x, enc_output):
        B,C,H,W=x.shape

        x=self.FT(x)
        out= self.Att1(x,x,x,c=C,h=H,w=W)
        x=self.layer_norm1(out+x)

        enc_output=self.FT(enc_output)
        out= self.Att2(enc_output, enc_output, x,c=C,h=H,w=W)
        x = self.layer_norm2(out + x)

        x = self.MLP(x).permute(0,2,1).view(B,C,H,W)

        return x

class Rot_Transformer_Encoder_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_encoder=6,num_decoder=6,num_heads=8,drop=0.0):
        super(Rot_Transformer_Encoder_Decoder, self).__init__()

        self.num_encoder=num_encoder
        self.num_decoder=num_decoder

        self.Encorder_list = nn.ModuleList([])
        for i in range(num_encoder):
            self.Encorder_list.append(Rot_Transformer_Encoder(in_channel=in_channel,num_heads=num_heads,drop=drop,h=h,w=w))

        self.Decorder_list = nn.ModuleList([])
        for i in range(num_decoder):
            self.Decorder_list.append(Rot_Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop,h=h,w=w))

    def forward(self, x):
        y=x
        for i in range (self.num_encoder):
            y=self.Encorder_list[i](y)
        for i in range (self.num_decoder):
            x=self.Decorder_list[i](x,y)
        return x

#####################################################

######################Standard_transformer Position###############

def Standard_Position_Encoding(max_len,d_model):
    pe=torch.zeros(max_len,d_model)
    pos=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
    div=torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
    pe[:,0::2]=torch.sin(pos*div)
    pe[:,1::2]=torch.cos(pos*div)
    return pe.permute(1,0)

class Standard_Position_Transformer_Encoder(nn.Module):
    def __init__(self,in_channel,h,w,num_heads=8,drop=0.0):
        super(Standard_Position_Transformer_Encoder,self).__init__()
        self.FT=Flatten_Transpose()
        self.Pos_Encoding=Standard_Position_Encoding(max_len=in_channel,d_model=h*w)
        self.Att = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True,
                                         add_bias_kv=False,
                                         add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None,
                                         dtype=None)
        self.layer_norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.MLP = PositionwiseFeedForward(d_in=in_channel, d_hid=in_channel * 4, dropout=drop)

    def forward(self, x):
        B,C,H,W=x.shape
        Pos=self.Pos_Encoding[0:H*W,0:C]

        if Pos.device!=x.device:
            Pos=Pos.to(x.device)

        x=self.FT(x)+Pos#[B,HW,C]
        out, _ = self.Att(x, x, x, need_weights=False)
        out = self.layer_norm(out + x)
        out = self.MLP(out).permute(0, 2, 1).view(B, C, H, W)

        return out

class Standard_Position_Transformer_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_heads=8,drop=0.0):
        super(Standard_Position_Transformer_Decoder, self).__init__()

        self.Pos_Encoding = Standard_Position_Encoding(max_len=in_channel, d_model=h * w)

        self.Att1 = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True,
                                         add_bias_kv=False,
                                         add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None,
                                         dtype=None)
        self.layer_norm1= nn.LayerNorm(in_channel, eps=1e-6)

        self.Att2 = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads, dropout=drop, bias=True,
                                          add_bias_kv=False,
                                          add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None,
                                          dtype=None)
        self.layer_norm2= nn.LayerNorm(in_channel, eps=1e-6)

        self.MLP = PositionwiseFeedForward(d_in=in_channel, d_hid=in_channel * 4, dropout=drop)

        self.FT=Flatten_Transpose()

    def forward(self, x, enc_output):
        B,C,H,W=x.shape

        Pos = self.Pos_Encoding[ 0:H * W,0:C]

        if Pos.device != x.device:
            Pos = Pos.to(x.device)

        x=self.FT(x)+Pos
        out, _ = self.Att1(x, x, x, need_weights=False)
        x = self.layer_norm1(out + x)

        enc_output = self.FT(enc_output)
        out, _ = self.Att2(enc_output, enc_output, x, need_weights=False)
        x = self.layer_norm2(out + x)

        x = self.MLP(x).permute(0, 2, 1).view(B, C, H, W)

        return x

class Standard_Position_Transformer_Encoder_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_encoder=6,num_decoder=6,num_heads=8,drop=0.0):
        super(Standard_Position_Transformer_Encoder_Decoder, self).__init__()

        self.num_encoder=num_encoder
        self.num_decoder=num_decoder

        self.Encorder_list = nn.ModuleList([])
        for i in range(num_encoder):
            self.Encorder_list.append(Standard_Position_Transformer_Encoder(in_channel=in_channel,num_heads=num_heads,drop=drop,h=h,w=w))

        self.Decorder_list = nn.ModuleList([])
        for i in range(num_decoder):
            self.Decorder_list.append(Standard_Position_Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop,h=h,w=w))

    def forward(self, x):
        y=x
        for i in range (self.num_encoder):
            y=self.Encorder_list[i](y)
        for i in range (self.num_decoder):
            x=self.Decorder_list[i](x,y)
        return x

##################################################

#######################Once-position_transformer############

class Once_Rot_Transformer_Encoder_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_encoder=6,num_decoder=0,num_heads=8,drop=0.0):
        super(Once_Rot_Transformer_Encoder_Decoder, self).__init__()

        self.num_encoder=num_encoder
        self.num_decoder=num_decoder

        self.Encorder_list = nn.ModuleList([])
        if num_encoder>=1:
            self.Encorder_list.append(Rot_Transformer_Encoder(in_channel=in_channel, num_heads=num_heads, drop=drop, h=h, w=w))
            for i in range(1,num_encoder):
                self.Encorder_list.append(Transformer_Encoder(in_channel=in_channel,num_heads=num_heads,drop=drop))

        self.Decorder_list = nn.ModuleList([])
        if num_decoder >= 1:
            self.Decorder_list.append(Rot_Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop, h=h, w=w))
        for i in range(1,num_decoder):
            self.Decorder_list.append(Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop))

    def forward(self, x):
        y=x
        for i in range (self.num_encoder):
            y=self.Encorder_list[i](y)
        for i in range (self.num_decoder):
            x=self.Decorder_list[i](x,y)
        return x

class Once_Standard_Position_Transformer_Encoder_Decoder(nn.Module):

    def __init__(self, in_channel,h,w,num_encoder=6,num_decoder=0,num_heads=8,drop=0.0):
        super(Once_Standard_Position_Transformer_Encoder_Decoder, self).__init__()

        self.num_encoder=num_encoder
        self.num_decoder=num_decoder

        self.Encorder_list = nn.ModuleList([])
        if num_encoder >= 1:
            self.Encorder_list.append(
                Standard_Position_Transformer_Encoder(in_channel=in_channel, num_heads=num_heads, drop=drop, h=h, w=w))
            for i in range(1, num_encoder):
                self.Encorder_list.append(Transformer_Encoder(in_channel=in_channel, num_heads=num_heads, drop=drop))

        self.Decorder_list = nn.ModuleList([])
        if num_decoder >= 1:
            self.Decorder_list.append(
                Standard_Position_Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop, h=h, w=w))
        for i in range(1, num_decoder):
            self.Decorder_list.append(Transformer_Decoder(in_channel=in_channel, num_heads=num_heads, drop=drop))

    def forward(self, x):
            y=x
            for i in range (self.num_encoder):
                y=self.Encorder_list[i](y)
            for i in range (self.num_decoder):
                x=self.Decorder_list[i](x,y)
            return x

########################################################################

class Rot_Selecting_Weight_Short_ED(nn.Module):

    def __init__(self, h,w,out_channel=7,drop=0.0,num_encoder=6,num_decoder=0,bn=False,mode='bicubic'):
        super(Rot_Selecting_Weight_Short_ED, self).__init__()
        self.mode=mode

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        self.trans4 = Rot_Transformer_Encoder_Decoder(in_channel=256, h=h//8,w=w//8,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        self.trans5 = Rot_Transformer_Encoder_Decoder(in_channel=512, h=h//16,w=w//16,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)

        x2 = self.down1(x)
        x3=self.down2(x2)
        x4 = self.trans4(self.down3(x3))
        y = self.trans5(self.down4(x4))

        y=(self.up5(y)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Once_Rot_Selecting_Weight_Short_ED(nn.Module):

    def __init__(self, h,w,out_channel=7,drop=0.0,num_encoder=6,num_decoder=0,bn=False,mode='bicubic'):
        super(Once_Rot_Selecting_Weight_Short_ED, self).__init__()
        self.mode=mode

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        self.trans4 = Once_Rot_Transformer_Encoder_Decoder(in_channel=256, h=h//8,w=w//8,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        self.trans5 = Once_Rot_Transformer_Encoder_Decoder(in_channel=512, h=h//16,w=w//16,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)

        x2 = self.down1(x)
        x3=self.down2(x2)
        x4 = self.trans4(self.down3(x3))
        y = self.trans5(self.down4(x4))

        y=(self.up5(y)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Standard_Position_Selecting_Weight_Short_ED(nn.Module):

    def __init__(self, h,w,out_channel=7,drop=0.0,num_encoder=6,num_decoder=0,bn=False,mode='bicubic'):
        super(Standard_Position_Selecting_Weight_Short_ED, self).__init__()
        self.mode=mode

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        self.trans4 = Standard_Position_Transformer_Encoder_Decoder(in_channel=256, h=h//8,w=w//8,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        self.trans5 = Standard_Position_Transformer_Encoder_Decoder(in_channel=512, h=h//16,w=w//16,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)

        x2 = self.down1(x)
        x3=self.down2(x2)
        x4 = self.trans4(self.down3(x3))
        y = self.trans5(self.down4(x4))

        y=(self.up5(y)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Once_Standard_Position_Selecting_Weight_Short_ED(nn.Module):

    def __init__(self, h,w,out_channel=7,drop=0.0,num_encoder=6,num_decoder=0,bn=False,mode='bicubic'):
        super(Once_Standard_Position_Selecting_Weight_Short_ED, self).__init__()
        self.mode=mode

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        self.trans4 = Once_Standard_Position_Transformer_Encoder_Decoder(in_channel=256, h=h//8,w=w//8,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        self.trans5 = Once_Standard_Position_Transformer_Encoder_Decoder(in_channel=512, h=h//16,w=w//16,num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)

        x2 = self.down1(x)
        x3=self.down2(x2)
        x4 = self.trans4(self.down3(x3))
        y = self.trans5(self.down4(x4))

        y=(self.up5(y)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

############################################################

####################Other ES structure####################

class Selecting_Weight_Uptrans_ED(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=3,num_decoder=3):
        super(Selecting_Weight_Uptrans_ED, self).__init__()
        self.mode=mode

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        self.trans4 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        self.trans5 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=128, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(64)
            self.bnlast_2 = nn.BatchNorm2d(128)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)

        x2 = self.down1(x)
        x3=self.down2(x2)
        x4 = self.down3(x3)
        y = self.trans5(self.down4(x4))

        y=(self.trans4(self.up5(y))+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Selecting_Weight_ED_Num_Double_512(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2):
        super(Selecting_Weight_ED_Num_Double_512, self).__init__()
        self.mode=mode
        self.trans_level=trans_level

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        if trans_level>=5:
            self.trans1 = Transformer_Encoder_Decoder(in_channel=64, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down1=Downsample(in_channel=64,out_channel=128,bn=bn)
        if trans_level>=4:
            self.trans2 = Transformer_Encoder_Decoder(in_channel=128, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down2=Downsample(in_channel=128,out_channel=256,bn=bn)
        if trans_level>=3:
            self.trans3 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down3=Downsample(in_channel=256,out_channel=512,bn=bn)
        if trans_level>=2:
            self.trans4 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=512,out_channel=1024,bn=bn)
        if trans_level>=1:
            self.trans5 = Transformer_Encoder_Decoder(in_channel=1024, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=1024,out_channel=512,bn=bn)
        self.up4 = Upsample(in_channel=512, out_channel=256,bn=bn)
        self.up3 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up2 = Upsample(in_channel=128, out_channel=64,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(64)
            self.bnlast_1 = nn.BatchNorm2d(256)
            self.bnlast_2 = nn.BatchNorm2d(512)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)
        if self.trans_level>=5:
            x = self.trans1(x)

        x2 = self.down1(x)
        if self.trans_level>=4:
            x2 = self.trans2(x2)

        x3=self.down2(x2)
        if self.trans_level>=3:
            x3 = self.trans3(x3)

        x4=self.down3(x3)
        if self.trans_level>=2:
            x4 = self.trans4(x4)

        x5=self.down4(x4)
        if self.trans_level>=1:
            x5 = self.trans5(x5)

        y=(self.up5(x5)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Selecting_Weight_ED_Num_512(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2):
        super(Selecting_Weight_ED_Num_512, self).__init__()
        self.mode=mode
        self.trans_level=trans_level

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        if trans_level>=5:
            self.trans1 = Transformer_Encoder_Decoder(in_channel=32, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)
        if trans_level>=4:
            self.trans2 = Transformer_Encoder_Decoder(in_channel=64, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)
        if trans_level>=3:
            self.trans3 = Transformer_Encoder_Decoder(in_channel=128, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        if trans_level>=2:
            self.trans4 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        if trans_level>=1:
            self.trans5 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(128)
            self.bnlast_2 = nn.BatchNorm2d(512)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)
        if self.trans_level>=5:
            x = self.trans1(x)

        x2 = self.down1(x)
        if self.trans_level>=4:
            x2 = self.trans2(x2)

        x3=self.down2(x2)
        if self.trans_level>=3:
            x3 = self.trans3(x3)

        x4=self.down3(x3)
        if self.trans_level>=2:
            x4 = self.trans4(x4)

        x5=self.down4(x4)
        if self.trans_level>=1:
            x5 = self.trans5(x5)

        y=(self.up5(x5)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

class Selecting_Weight_ED_Num_256(nn.Module):

    def __init__(self, out_channel=7,drop=0.0,bn=False,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2):
        super(Selecting_Weight_ED_Num_256, self).__init__()
        self.mode=mode
        self.trans_level=trans_level

        self.conv1_1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,stride=1)
        self.conv1_2= nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        if trans_level>=5:
            self.trans1 = Transformer_Encoder_Decoder(in_channel=32, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down1=Downsample(in_channel=32,out_channel=64,bn=bn)
        if trans_level>=4:
            self.trans2 = Transformer_Encoder_Decoder(in_channel=64, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down2=Downsample(in_channel=64,out_channel=128,bn=bn)
        if trans_level>=3:
            self.trans3 = Transformer_Encoder_Decoder(in_channel=128, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down3=Downsample(in_channel=128,out_channel=256,bn=bn)
        if trans_level>=2:
            self.trans4 = Transformer_Encoder_Decoder(in_channel=256, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.down4=Downsample(in_channel=256,out_channel=512,bn=bn)
        if trans_level>=1:
            self.trans5 = Transformer_Encoder_Decoder(in_channel=512, num_encoder=num_encoder, num_decoder=num_decoder, num_heads=8, drop=drop)

        self.up5=Upsample(in_channel=512,out_channel=256,bn=bn)
        self.up4 = Upsample(in_channel=256, out_channel=128,bn=bn)
        self.up3 = Upsample(in_channel=128, out_channel=64,bn=bn)
        self.up2 = Upsample(in_channel=64, out_channel=32,bn=bn)

        self.conv_last_1=nn.Conv2d(in_channels=32,out_channels=128,kernel_size=3,padding=1,stride=1)
        self.conv_last_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv_last_3 = nn.Conv2d(in_channels=256, out_channels=out_channel, kernel_size=1, padding=0, stride=1)

        if bn:
            self.bn1_1=nn.BatchNorm2d(32)
            self.bn1_2 = nn.BatchNorm2d(32)
            self.bnlast_1 = nn.BatchNorm2d(128)
            self.bnlast_2 = nn.BatchNorm2d(256)
        self.bn=bn

        self.act = nn.LeakyReLU()

        self.apply(weight_init)

        self.gamma1=nn.Parameter(torch.Tensor([1]))
        self.gamma2 = nn.Parameter(torch.Tensor([1]))
        self.gamma3 = nn.Parameter(torch.Tensor([1]))
        self.gamma4 = nn.Parameter(torch.Tensor([1]))

    def forward(self, x):

        x=self.conv1_1(x)
        if self.bn:
            x=self.bn1_1(x)
        x=self.conv1_2(x)
        if self.bn:
            x=self.bn1_2(x)
        x = self.act(x)
        if self.trans_level>=5:
            x = self.trans1(x)

        x2 = self.down1(x)
        if self.trans_level>=4:
            x2 = self.trans2(x2)

        x3=self.down2(x2)
        if self.trans_level>=3:
            x3 = self.trans3(x3)

        x4=self.down3(x3)
        if self.trans_level>=2:
            x4 = self.trans4(x4)

        x5=self.down4(x4)
        if self.trans_level>=1:
            x5 = self.trans5(x5)

        y=(self.up5(x5)+self.gamma4*x4)/(1+self.gamma4)
        y = (self.up4(y) + self.gamma3*x3)/(1+self.gamma3)
        y = (self.up3(y) + self.gamma2*x2)/(1+self.gamma2)
        y = (self.up2(y) + self.gamma1*x)/(1+self.gamma1)

        y=self.conv_last_1(y)
        if self.bn:
            y=self.bnlast_1(y)
        y=self.act(y)

        y=self.conv_last_2(y)
        if self.bn:
            y=self.bnlast_2(y)
        y=self.act(y)

        y = F.softmax(self.conv_last_3(y),dim=1)

        return y

if __name__ == '__main__':

    Crop = [320, 320]
    Test=Selecting_Weight_ED_Num_Double(trans_level=2)
    get_parameter_number(Test)
    x=torch.rand(4,3,32,32)
    y=Test(x)
    print(y.shape)
