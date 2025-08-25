import numpy as np
import torch
from torchvision import transforms as T
import os
import cv2

def Torch_Normalizion(input):
    return input/input.max()

def Predict_ES_list(Extractor_list,Selector_list,Data):

    assert (len(Selector_list) == 0 and len(Extractor_list) == 1) or (len(Selector_list) > 0 and len(Extractor_list)>0)

    if len(Selector_list) == 0:
        F_Pre = Extractor_list[0](Data)[0]
        if type(F_Pre)==type([]):
            Out =F_Pre[-1]
        else:
            Out=F_Pre

    else:
        B,C,H,W=Data.shape

        Pre_list=[]
        for j in range(len(Extractor_list)):
            Pre_list.append(Extractor_list[j](Data)[1])
        F_Pre =torch.cat(Pre_list,dim=1)

        # F_Pre = Extractor_list[0](Data)[1]
        # for j in range (1,len(Extractor_list)):
        #     F_Pre = torch.cat([F_Pre,Extractor_list[j](Data)[1]],dim=1)

        Out = F_Pre * Selector_list[0](Data)

        if len(Selector_list)>1:
            Total_zero = torch.zeros(B, 1, H, W).to(Data.device)
            Total_one = torch.ones(B, 1, H, W).to(Data.device)
            for j in range(1, len(Selector_list)):
                Out = Torch_Normalizion(Out)
                Out = torch.cat([Total_zero, Out, Total_one], dim=1)
                Out = Out * Selector_list[j](Data)

        Out = Out.sum(1).unsqueeze(1).clamp(0, 1)

    return Out

def Cat_avg(Matrix_list,over,over_last_H,over_last_W):#list[h][w]
    Num_H=len(Matrix_list)
    Num_W=len(Matrix_list[0])
    Row_list=[]
    if Num_W>1:
        for i in range (Num_H):
            Row=Matrix_list[i][0]
            for j in range (1,Num_W-1):
                left=Row[:,0:-over]
                middle=(Row[:,-over:]+Matrix_list[i][j][:,:over])/2
                right=Matrix_list[i][j][:,over:]
                Row=torch.cat([left,middle,right],dim=1)
            left = Row[:, 0:-over_last_W]
            middle = (Row[:, -over_last_W:] + Matrix_list[i][-1][:, :over_last_W]) / 2
            right = Matrix_list[i][-1][:, over_last_W:]
            Row=torch.cat([left,middle,right],dim=1)
            Row_list.append(Row)
    else:
        for i in range (Num_H):
            Row_list.append(Matrix_list[i][0])

    Out = Row_list[0]
    if len(Row_list) > 1:
        for i in range(1,len(Row_list)-1):
            up = Out[0:-over, :]
            middle = (Out[ -over:,:] + Row_list[i][ :over,:]) / 2
            down = Row_list[i][ over:,:]
            Out = torch.cat([up, middle, down], dim=0)
        up = Out[ 0:-over_last_H,:]
        middle = (Out[-over_last_H:, :] + Row_list[-1][:over_last_H,: ]) / 2
        down = Row_list[-1][ over_last_H:,:]
        Out = torch.cat([up, middle, down], dim=0)

    return Out

def Save_Predict_ES_list(Extractor_list,Selector_list, Data_dir, Save_pred_dir,device='cpu',pred_require=16,block=None,over=16):

    assert len(Extractor_list)>0

    for i in range (len(Extractor_list)):
        Extractor_list[i].to(device)
        Extractor_list[i].eval()

    for i in range (len(Selector_list)):
        Selector_list[i].to(device)
        Selector_list[i].eval()

    Data_name = os.listdir(Data_dir)
    Data_name.sort()

    if block:
        assert len(block)==2
        Block_H,Block_W=block

    for index in range(len(Data_name)):
        Data_file = os.path.join(Data_dir, Data_name[index])
        Data = cv2.imread(Data_file, cv2.IMREAD_COLOR)
        Data = cv2.cvtColor(Data, cv2.COLOR_BGR2RGB)

        h,w,c=Data.shape

        h_r=h//pred_require*pred_require
        w_r=w//pred_require*pred_require

        if h!=h_r or w!=w_r:
            Data=cv2.resize(Data,(w_r,h_r))

        Data=T.ToTensor()(Data)
        Data=normalize(Data).unsqueeze(0)

        over_last_W = over
        over_last_H = over

        if block:
            Data_H,Data_W=h_r,w_r
            Data_list=[]

            if Data_H > Block_H:
                Data_list = [Data[:, :, :Block_H, :]]
                for i in range(1, (Data_H - Block_H - 1) // (Block_H - over) + 1):
                    Data_list.append(Data[:, :, i * (Block_H - over):i * (Block_H - over) + Block_H, :])
                Data_list.append(Data[:, :, Data_H - Block_H:Data_H,: ])
                over_last_H = Block_H - (Data_H - ((Data_H - Block_H - 1) // (Block_H - over) * (Block_H - over) + Block_H))

            else:
                Data_list.append(Data)

            if Data_W>Block_W:
               for i in range (len(Data_list)):
                   Data_W_list = [Predict_ES_list(Extractor_list,Selector_list,Data=Data_list[i][:, :, :, :Block_W].to(device)).squeeze().data.cpu()]
                   for j in range(1, (Data_W - Block_W - 1) // (Block_W - over) + 1):
                       Data_W_list.append(Predict_ES_list(Extractor_list,Selector_list,Data=Data_list[i][:, :, :, j*(Block_W-over):j*(Block_W-over)+Block_W].to(device)).squeeze().data.cpu())
                   over_last_W = Block_W - (Data_W - ((Data_W - Block_W - 1) // (Block_W - over) * (Block_W - over) + Block_W))
                   Data_W_list.append(Predict_ES_list(Extractor_list,Selector_list,Data=Data_list[i][:,:,:,Data_W-Block_W:Data_W].to(device)).squeeze().data.cpu())
                   Data_list[i]=Data_W_list

            else:
                for i in range(len(Data_list)):
                    Data_list[i] = [Predict_ES_list(Extractor_list,Selector_list,Data=Data_list[i].to(device)).squeeze().data.cpu()]

            Out=Cat_avg(Matrix_list=Data_list,over=over,over_last_H=over_last_H,over_last_W=over_last_W)

        else:
            Out = Predict_ES_list(Extractor_list,Selector_list,Data=Data.to(device)).squeeze().data.cpu()

        Out = Out.numpy()

        if h != h_r or w != w_r:
            Out = cv2.resize(Out, (w, h)).astype(np.float32)

        Out = Unit_Normalize(Out)
        cv2.imwrite(os.path.join(Save_pred_dir, Data_name[index][:-4] + '.png'), Out)

        torch.cuda.empty_cache()

if __name__ == '__main__':

########### Use the codes to save predictions ###########

    from ES_EES_SWBCE_EBT_BAA import *

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'
    print(device)

    # device='cpu'
    from Dexi_Edge_Select import *
    Extractor_1= Dexi_Edge_EES_Extractor(device=device)
    Extractor_1 = nn.DataParallel(Extractor_1)
    Extractor_1.load_state_dict(torch.load('',map_location=device)) # Checkpoint path
    Extractor_1 = Extractor_1.to(device)

    from BDCN_Edge_Select import *
    Extractor_2 = BDCN_Edge_EES_Extractor(device=device)
    Extractor_2 = nn.DataParallel(Extractor_2)
    Extractor_2.load_state_dict(torch.load('',map_location=device))# Checkpoint path
    Extractor_2 = Extractor_2.to(device)

    from HED_Edge_Select import *
    Extractor_3 = HED_Edge_EES_Extractor(device=device)
    Extractor_3 = nn.DataParallel(Extractor_3)
    Extractor_3.load_state_dict(torch.load('',map_location=device))# Checkpoint path
    Extractor_3 = Extractor_3.to(device)

    Selector_1 = Selecting_Weight_ED_Num_Double(out_channel=447, drop=0.0, bn=True, mode='bicubic', num_encoder=6,num_decoder=0, trans_level=2)# Selector model
    Selector_1 = nn.DataParallel(Selector_1)
    Selector_1.load_state_dict(torch.load('', map_location=device))# Checkpoint path
    Selector_1 = Selector_1.to(device)

    # Selector_2 = Selecting_Weight_ED_Num_Double(out_channel=449, drop=0.0, bn=True, mode='bicubic', num_encoder=6,num_decoder=0, trans_level=2)
    # Selector_2 = nn.DataParallel(Selector_2)
    # Selector_2.load_state_dict(torch.load('', map_location=device))# Checkpoint path
    # Selector_2 = Selector_2.to(device)

    Extractor_list = [Extractor_1,Extractor_2,Extractor_3]
    Selector_list = [Selector_1] # [Selector_1,Selector_2]

    Save_Predict_ES_list(Extractor_list,Selector_list,Data_dir='',Save_pred_dir='',block=[320,320],over=16,pred_require=1,device=device)
    # Images will be resized to n*pred_require (both height and width) and predicted piece by piece (block), then cat with overlap (over)
    # Set the data path and save path above
