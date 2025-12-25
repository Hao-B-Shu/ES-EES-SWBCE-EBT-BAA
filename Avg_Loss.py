
import numpy as np
from DataLoad import *
from torch.utils.data import DataLoader
from ES_EES_SWBCE_EBT_BAA import SWBCE
from EdgeNAT_Select import *
from Predict_Final import Torch_Normalizion

def Avg_Loss(loss,Output_dir,Label_dir,Extractor_list=[],Selector_list=[],crop=[320,320],pred_require=16,batch=8,shuffle=True,num_work=8,device='cpu'):

    Train_data = Generate_Dataset(data_path=Output_dir, label_path=Label_dir, test_or_not=False,
                                  crop_size=crop, pred_require=pred_require)
    dataloader = DataLoader(Train_data, batch_size=batch, shuffle=shuffle, num_workers=num_work)
    Loss_list = []
    assert len(Extractor_list)>0

    if len(Selector_list)==0:
        assert len(Extractor_list)==1
        Extractor = Extractor_list[0].to(device)
        Extractor.eval()
        for i, data in enumerate(dataloader):
            image = data['image'].to(device)
            label = data['label'].to(device)
            pred = Extractor(image)[0]
            current_loss = loss(pred, label)
            Loss_list.append(current_loss.item())
    else:
        for i in range (len(Extractor_list)):
            Extractor_list[i]=Extractor_list[i].to(device)
            Extractor_list[i].eval()
        for i in range (len(Selector_list)):
            Selector_list[i]=Selector_list[i].to(device)
            Selector_list[i].eval()

        for i, data in enumerate(dataloader):
            image = data['image'].to(device)
            label = data['label'].to(device)

            Output_list = []
            for j in range(0, len(Extractor_list)):
                Output_list.append(Extractor_list[j](image)[1])
            Feature = torch.cat(Output_list, dim=1)

            Weight = Selector_list[0](image)

            Total_zero = torch.zeros_like(label).to(device)
            Total_one = torch.ones_like(label).to(device)
            Pred = Feature * Weight

            for j in range(1, len(Selector_list)):
                Pred = Torch_Normalizion(Pred)
                Pred = torch.cat([Total_zero, Pred, Total_one], dim=1)
                Pred = Pred * Selector_list[j](image)
            pred = Pred.sum(1).unsqueeze(1).clamp(0, 1)
            current_loss = loss(pred, label)
            Loss_list.append(current_loss.item())
    return np.array(Loss_list).mean()
####################################################################


if __name__=='__main__':

    print('Processing')

    #############################To calculate the average loss#############################

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'

    data_dir = '' # Set the path of the data dir here
    Label_dir='' # Set the path of the label dir here
    Extractor_list=[EdgeNAT_ES_Extraction()] # Cannot be []
    Selector_list=[] # Set [] if no selector
    Loss = SWBCE(Label_Pred_balance=1) # Set loss
    L=Avg_Loss(loss=Loss,Output_dir=data_dir,Label_dir=Label_dir,Extractor_list=Extractor_list,Selector_list=Selector_list,device=device)
    print(L)
    ###############################################################################################
