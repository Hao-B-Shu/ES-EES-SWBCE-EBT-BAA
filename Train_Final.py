import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from datetime import datetime
import os

def Torch_Normalizion(input):
    return input/input.max()

def single_train(dataloader, device='cpu',mode=None,level=0):

    global Extractor_list,Selector_list,Loss_Extractor_list,Loss_Selector_list,Optimizer_Extractor_list,Optimizer_Selector_list

    assert mode=='Extractor' or mode=='Selector' or mode=='Union'
    assert level>=0

    if mode=='Extractor':

        loss = []
        torch.cuda.empty_cache()
        for i, data in enumerate(dataloader):
            image = data['image'].to(device)
            label = data['label'].to(device)
            pred = Extractor_list[level](image)[0]
            current_loss = Loss_Extractor_list[level](pred, label)
            Optimizer_Extractor_list[level].zero_grad()
            current_loss.backward()
            Optimizer_Extractor_list[level].step()

            loss.append(current_loss.item())
            # if i % 100 == 0:
            #     print(datetime.now(), str(i), np.array(loss).mean())
        return np.array(loss).mean()

    if mode=='Selector':

        if level==0:

            loss = []
            torch.cuda.empty_cache()
            for i, data in enumerate(dataloader):
                image = data['image'].to(device)
                label = data['label'].to(device)

                Output_list=[]
                for j in range (0,len(Extractor_list)):
                    Output_list.append(Extractor_list[j](image)[1])
                Feature = torch.cat(Output_list,dim=1)

                pred = (Feature*Selector_list[0](image)).sum(1).unsqueeze(1).clamp(0,1)
                current_loss = Loss_Selector_list[0](pred, label)
                Optimizer_Selector_list[0].zero_grad()
                current_loss.backward()
                Optimizer_Selector_list[0].step()

                loss.append(current_loss.item())
                # if i % 100 == 0:
                #     print(datetime.now(), str(i), np.array(loss).mean())
            return np.array(loss).mean()

        if level>0:

            loss = []
            torch.cuda.empty_cache()
            for i, data in enumerate(dataloader):
                image = data['image'].to(device)
                label = data['label'].to(device)

                Output_list=[]
                for j in range (0,len(Extractor_list)):
                    Output_list.append(Extractor_list[j](image)[1])
                Feature = torch.cat(Output_list,dim=1)

                Weight=Selector_list[0](image)

                Total_zero=torch.zeros_like(label).to(device)
                Total_one = torch.ones_like(label).to(device)
                Pred=Feature*Weight

                for j in range (1,level):
                    Pred=Torch_Normalizion(Pred)
                    Pred=torch.cat([Total_zero,Pred,Total_one],dim=1)
                    Pred=Pred*Selector_list[j](image)
                pred = Pred.sum(1).unsqueeze(1).clamp(0,1)
                current_loss = Loss_Selector_list[level](pred, label)
                Optimizer_Selector_list[level].zero_grad()
                current_loss.backward()
                Optimizer_Selector_list[level].step()

                loss.append(current_loss.item())
                # if i % 100 == 0:
                #     print(datetime.now(), str(i), np.array(loss).mean())
        return np.array(loss).mean()

    if mode == 'Union':
        assert level>0
        loss = []
        torch.cuda.empty_cache()

        for i, data in enumerate(dataloader):
            image = data['image'].to(device)
            label = data['label'].to(device)

            Output_list=[]
            for j in range(0, len(Extractor_list)):
                Output_list.append(Extractor_list[j](image)[1])
            Feature = torch.cat(Output_list,dim=1)

            Weight = Selector_list[0](image)
            Total_zero = torch.zeros_like(label).to(device)
            Total_one = torch.ones_like(label).to(device)
            Pred = Feature * Weight

            for j in range(1, level):
                Pred = Torch_Normalizion(Pred)
                Pred = torch.cat([Total_zero, Pred, Total_one], dim=1)
                Pred = Pred * Selector_list[j](image)
            Pred = Pred.sum(1).unsqueeze(1).clamp(0, 1)

            current_loss = Loss_Selector_list[level-1](Pred, label)
            for k in range(len(Optimizer_Extractor_list)):
                Optimizer_Extractor_list[k].zero_grad()
            for l in range(len(Optimizer_Selector_list)):
                Optimizer_Selector_list[l].zero_grad()

            current_loss.backward()
            for k in range(len(Optimizer_Extractor_list)):
                Optimizer_Extractor_list[k].step()
            for l in range(level):
                Optimizer_Selector_list[l].step()

            loss.append(current_loss.item())
            # if i % 100 == 0:
            #     print(datetime.now(), str(i), np.array(loss).mean())
        return np.array(loss).mean()

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

def Train(
        train_data_dir: str = 'Train/Data',
        train_label_dir: str = 'Train/Label',
        Pretrain_dir: str = 'Pretrain_Model',
        device: str = 'cpu',
        epoch: int = 0,
        epoch_Extractor_list: list=[],
        epoch_Selector_list: list=[],
        refresh: int = 2,
        print_loss: int=1,
        save_per_epoch: int = 5,
        lr_decrease: any=None,
        train_crop=None,
        train_batch=1,
        train_shuffle=True,
        num_work=8,
        pred_require=16,
        Union=None
):

    global Extractor_list,Selector_list,Loss_Extractor_list,Loss_Selector_list,Optimizer_Extractor_list,Optimizer_Selector_list,Extractor_Name,Selector_Name

    torch.manual_seed(50*epoch+100)
    Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir, test_or_not=False,
                                  crop_size=train_crop,pred_require=pred_require)
    Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle, num_workers=num_work)

    Max_Extractor_index=min([len(epoch_Extractor_list),len(Extractor_list),len(Loss_Extractor_list),len(Optimizer_Extractor_list)])
    Max_Selector_index = min([len(epoch_Selector_list), len(Selector_list), len(Loss_Selector_list), len(Optimizer_Selector_list)])

    for i in range(Max_Extractor_index):
        Extractor_list[i].eval()

    for i in range (Max_Extractor_index):
        if epoch_Extractor_list[i]>0:
            Extractor_list[i].train()
            print('Train Extractor ' + str(i+1))
            get_parameter_number(Extractor_list[i])

        for Epoch in range (0,epoch_Extractor_list[i]):

            loss_list = []

            if (lr_decrease != None and Epoch % lr_decrease == 0 and Epoch != 0):
                for params in Optimizer_Extractor_list[i].param_groups:
                    params['lr'] *= 0.1
                    print(params['lr'])

            if Epoch % refresh == 0 and Epoch != 0:
                torch.manual_seed(50 * Epoch + epoch)
                Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir,
                                              test_or_not=False, crop_size=train_crop, pred_require=pred_require)
                Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle,
                                          num_workers=num_work)

            loss_avg = single_train(dataloader=Train_loader, device=device,mode='Extractor',level=i)
            loss_list.append(loss_avg)

            if (Epoch + 1) % save_per_epoch == 0:
                if len(Extractor_Name)<=i:
                    torch.save(Extractor_list[i].state_dict(),
                               Pretrain_dir + '/' + 'Extractor_'+str(i+1)+'epoch_' + str(Epoch + 1)+ '_loss_' + str(loss_list[-1])+ '.pth')
                else:
                    torch.save(Extractor_list[i].state_dict(),
                               Pretrain_dir + '/'  + Extractor_Name[i] + 'epoch_' + str(Epoch + 1) + '.pth')

            if (Epoch + 1) % print_loss == 0:
                print('epoch: ' + str(Epoch + 1))
                print('loss=' + str(loss_list[-1]))
                print(datetime.now())

        epoch=epoch+epoch_Extractor_list[i]
        Extractor_list[i].eval()

    for i in range(Max_Selector_index):
        Selector_list[i].eval()

    for i in range (Max_Selector_index):
        Selector_list[i].train()

        if epoch_Selector_list[i]>0:
            print('Train Selector ' + str(i+1))
            get_parameter_number(Selector_list[i])

        for Epoch in range (0,epoch_Selector_list[i]):

            loss_list = []

            if (lr_decrease != None and Epoch % lr_decrease == 0 and Epoch != 0):
                for params in Optimizer_Selector_list[i].param_groups:
                    params['lr'] *= 0.1
                    print(params['lr'])

            if Epoch % refresh == 0 and Epoch != 0:
                torch.manual_seed(50 * Epoch + epoch)
                Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir,
                                              test_or_not=False, crop_size=train_crop, pred_require=pred_require)
                Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle,
                                          num_workers=num_work)

            loss_avg = single_train(dataloader=Train_loader, device=device,mode='Selector',level=i)
            loss_list.append(loss_avg)

            if (Epoch + 1) % save_per_epoch == 0:
                  if len(Selector_Name)>0:
                        if len(Selector_Name)<=i:
                            torch.save(Selector_list[i].state_dict(),
                                   Pretrain_dir + '/' + 'Selector_'+str(i+1)+'epoch_' + str(Epoch + 1) + '_loss_' + str(loss_list[-1])+ '.pth')
                        else:
                            torch.save(Selector_list[i].state_dict(),
                                       Pretrain_dir + '/' + Selector_Name[i] + 'epoch_' + str(Epoch + 1)  + '.pth')
                  else:
                      torch.save(Selector_list[i].state_dict(),
                                     Pretrain_dir + '/' + 'Selector_' + str(i + 1) + 'epoch_' + str(
                                         Epoch + 1) + '_loss_' + str(loss_list[-1]) + '.pth')

            if (Epoch + 1) % print_loss == 0:
                print('epoch: ' + str(Epoch + 1))
                print('loss=' + str(loss_list[-1]))
                print(datetime.now())

        epoch = epoch + epoch_Selector_list[i]
        Selector_list[i].eval()

    if Union!=None:
        Max_Extractor_index = min([ len(Extractor_list), len(Loss_Extractor_list), len(Optimizer_Extractor_list)])
        Select_level = min([len(Selector_list), len(Loss_Selector_list), len(Optimizer_Selector_list)])

        for i in range(Max_Extractor_index):
            print('Train Union Extractor '+ str(i+1))
            Extractor_list[i].train()
            get_parameter_number(Extractor_list[i])

        for i in range(Select_level):
            print('Train Union Selector ' + str(i + 1))
            Selector_list[i].train()
            get_parameter_number(Selector_list[i])

        print('Union optimizing with '+str(Max_Extractor_index)+' extractors and '+str(Select_level)+' selectors')

        for Epoch in range(0, Union):

            loss_list = []

            if (lr_decrease != None and Epoch % lr_decrease == 0 and Epoch != 0):
                for i in range(Max_Extractor_index):
                    for params in Optimizer_Extractor_list[i].param_groups:  # Search every para in Optimizer
                        params['lr'] *= 0.1
                        print(params['lr'])
                for i in range(Select_level):
                    for params in Optimizer_Selector_list[i].param_groups:
                        params['lr'] *= 0.1
                        print(params['lr'])

            if Epoch % refresh == 0 and Epoch != 0: # Refresh data per refresh
                torch.manual_seed(50 * Epoch + epoch)
                Train_data = Generate_Dataset(data_path=train_data_dir, label_path=train_label_dir,
                                              test_or_not=False, crop_size=train_crop, pred_require=pred_require)
                Train_loader = DataLoader(Train_data, batch_size=train_batch, shuffle=train_shuffle,
                                          num_workers=num_work)

            loss_avg = single_train(dataloader=Train_loader, device=device, mode='Union', level=Select_level)
            loss_list.append(loss_avg)

            if (Epoch + 1) % save_per_epoch == 0: # Save every save_per_epoch
                for i in range(Max_Extractor_index):
                    if len(Extractor_Name) <= i:
                        torch.save(Extractor_list[i].state_dict(),
                                   Pretrain_dir + '/' + 'Extractor_' + str(i + 1) + '_Union_epoch_' + str(
                                       Epoch + 1) + '_loss_' + str(loss_list[-1]) + '.pth')
                    else:
                        torch.save(Extractor_list[i].state_dict(),
                                   Pretrain_dir + '/'  + Extractor_Name[i] + '_Union_epoch_' + str(Epoch + 1)+ '.pth')
                for i in range(Select_level):
                    if len(Selector_Name) <= i:
                        torch.save(Selector_list[i].state_dict(),
                                   Pretrain_dir + '/' + 'Selector_' + str(i + 1) + '_Union_epoch_' + str(
                                       Epoch + 1) + '_loss_' + str(loss_list[-1]) + '.pth')
                    else:
                        torch.save(Selector_list[i].state_dict(),
                                   Pretrain_dir + '/' + Selector_Name[i] + '_Union_epoch_' + str(Epoch + 1) + '.pth')

            if (Epoch + 1) % print_loss == 0:
                print('epoch: ' + str(Epoch + 1))
                print('loss=' + str(loss_list[-1]))
                print(datetime.now())

        for i in range(Max_Extractor_index):
            Extractor_list[i].eval()
        for i in range(Select_level):
            Selector_list[i].eval()

#######The codes are for training models, set the parameters in the following and run##############

###################Set device and import##############
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
print(device)
from DataLoad import * #Dataloader
from ES_EES_SWBCE_EBT_BAA import * # Selector models and losses are in here
######################################################

#################Set Extractors#######################
from Dexi_Edge_Select import *
Extractor_1=Dexi_Edge_EES_Extractor(device=device)
Extractor_1_Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Extractor_1.parameters()), lr=0.0001, weight_decay=0.00000001)
Extractor_1 = nn.DataParallel(Extractor_1)
#Extractor_1.load_state_dict(torch.load('', map_location=device))
Extractor=Extractor_1.to(device)

from BDCN_Edge_Select import *
Extractor_2=BDCN_Edge_EES_Extractor(device=device)
Extractor_2_Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Extractor_2.parameters()), lr=0.0001, weight_decay=0.00000001)
Extractor_2 = nn.DataParallel(Extractor_2)
#Extractor_2.load_state_dict(torch.load('', map_location=device))
Extractor_2=Extractor_2.to(device)

from HED_Edge_Select import *
Extractor_3=HED_Edge_EES_Extractor(device=device)
Extractor_3_Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Extractor_3.parameters()), lr=0.0001, weight_decay=0.00000001)
Extractor_3 = nn.DataParallel(Extractor_3)
#Extractor_3.load_state_dict(torch.load('', map_location=device))
Extractor_3=Extractor_3.to(device)

Extractor_list=[Extractor_1,Extractor_2,Extractor_3]
Extractor_Name=['Dexi','BDCN','HED']
Loss_Extractor_list=[SWBCE(l_weight=[0.7,0.7,1.1,1.1,0.3,0.3,1.3],Label_Pred_balance=0),SWBCE(l_weight=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.1],Label_Pred_balance=0),SWBCE(l_weight=[1,1,1,1,1,1],Label_Pred_balance=0)]
Optimizer_Extractor_list=[Extractor_1_Optimizer,Extractor_2_Optimizer,Extractor_3_Optimizer]
epoch_Exactror_list=[1,1,1] # Extractor optimized epochs
############################################################


##############Set Selectors#############################
Selector_1=Selecting_Weight_ED_Num_Double(out_channel=447,drop=0.0,bn=True,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2)
Selector_1_Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Selector_1.parameters()), lr=0.0001, weight_decay=0.00000001)
Selector_1 = nn.DataParallel(Selector_1)
# Selector_1.load_state_dict(torch.load('', map_location=device))
Selector_1=Selector_1.to(device)

# The number of out channels of the first selector is the sum of the number of out channels of all used extractors for ES and additional +2 for EES
# The number of out channels for each level selectors should +2 compared with the previous one

# Selector_2=Selecting_Weight_ED_Num_Double(out_channel=449,drop=0.0,bn=True,mode='bicubic',num_encoder=6,num_decoder=0,trans_level=2)
# Selector_2_Optimizer = optim.Adam(filter(lambda p: p.requires_grad, Selector_1.parameters()), lr=0.0001, weight_decay=0.00000001)
# Selector_2 = nn.DataParallel(Selector_2)
# # Selector_1.load_state_dict(torch.load('', map_location=device))
# Selector_2=Selector_2.to(device)

Selector_list=[Selector_1] # [Selector_1,Selector_2]
Selector_Name=['Selector1'] # ['Selector1','Selector2']
Loss_Selector_list=[SWBCE(Label_Pred_balance=0)] # [SWBCE(),SWBCE()], Label_Pred_balance=0 represents the WBCE loss
Optimizer_Selector_list=[Selector_1_Optimizer] #[Selector_1_Optimizer,Selector_2_Optimizer]
epoch_Selector_list=[1], # [1,1], Selector optimized epochs
##################################################################


################################Main#############################
Crop=[320,320] # Size that images crop to

Train(
    train_data_dir='Train_Data', # Train data path
    train_label_dir='Train_Label', # Train label path
    Pretrain_dir= 'Test', # Save dir
    device=device,
    epoch=0, # Only for random-sample seed, epochs for optimization should be set in epoch_list
    epoch_Extractor_list=epoch_Exactror_list, # Extractor optimized epochs
    epoch_Selector_list=epoch_Selector_list, # Selector optimized epochs
    refresh=1, # Period epoch for refreshing data
    print_loss=1, # Period epoch for printing loss
    save_per_epoch= 1, # Period epoch for saving checkpoints
    lr_decrease= None, # Period epoch for decreasing learn rate to 1/10
    train_crop=Crop, # Size that images crop to
    train_batch= 8,
    train_shuffle= True,
    num_work= 8,
    pred_require= 16, # Ignore it, only valid in test but test is not used this code
    Union=None # Union optimizing epoch, set to a int (the epochs) for joint optimization or set to None otherwise
)

torch.cuda.empty_cache()
###########################################
