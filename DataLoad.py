from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms as T
import os
import cv2

class Generate_Dataset(Dataset):

    def __init__(self,
            data_path,
            label_path=None,
            test_or_not=True,
            crop_size=None,
            pred_require=16,
            device='cpu'
                 ):

        self.Data_path=data_path
        self.Label_path=label_path
        self.Test_or_not=test_or_not
        self.Data_name=os.listdir(self.Data_path)
        self.Data_name.sort()
        self.device=device
        self.totensor=T.ToTensor()# RGB [0,1] [3,h,w]
        self.normalize=T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        self.pred_require=pred_require

        if self.Label_path!=None:
            self.Label_name = os.listdir(self.Label_path)
            self.Label_name.sort()

        if test_or_not==False:
          print('Train mode')
        else:
          print('Test mode')

        if crop_size!=None:
            assert len(crop_size)==2
            assert crop_size[0]%pred_require==0
            assert crop_size[1] % pred_require == 0
            self.Crop_size=crop_size#[int，int]
        else:
          DataFile=os.path.join(self.Data_path, self.Data_name[0])
          CropTo=cv2.imread(DataFile, cv2.IMREAD_GRAYSCALE).shape
          self.Crop_size=[CropTo[0]//pred_require*pred_require,CropTo[1]//pred_require*pred_require]

    def __getitem__(self, index):

        Data_file=os.path.join(self.Data_path, self.Data_name[index])
        Data=cv2.imread(Data_file, cv2.IMREAD_COLOR)#[H,W,3],BGR,numpy
        Data = cv2.cvtColor(Data, cv2.COLOR_BGR2RGB)#RGB [H,W,3] numpy
        Data=self.totensor(Data).float().to(self.device)#RGB Tensor[3,H,W], [0,1]
        Data=self.normalize(Data)
        C,H,W=Data.shape
        if self.Test_or_not==True and self.Label_path==None:
            Data=Data[:,0:H//self.pred_require*self.pred_require,0:W//self.pred_require*self.pred_require]
            return dict(image=Data, image_name=self.Data_name[index])#Tensor[3,H,W]
        elif self.Test_or_not==True and self.Label_path!=None:
            Label_file = os.path.join(self.Label_path, self.Label_name[index])
            Label = cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)  # [H,W],numpy
            Label = self.totensor(Label).float().to(self.device)  # Tensor[1,H,W]
            Data = Data[:, 0:H // self.pred_require * self.pred_require, 0:W // self.pred_require * self.pred_require]
            Label = Label[:, 0:H // self.pred_require * self.pred_require, 0:W // self.pred_require * self.pred_require]
            return dict(image=Data, image_name=self.Data_name[index],label=Label)
        else:
          Label_file=os.path.join(self.Label_path, self.Label_name[index])
          Label=cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)#[H,W],numpy
          Label=self.totensor(Label).float().to(self.device)#Tensor[1,H,W]
          [Crop_data,Crop_label]=self.Crop(Data,Label)
          return dict(image=Crop_data,label=Crop_label)#Tensor[C,H,W]

    def __len__(self):

        return len(self.Data_name)

    def Crop(self,img,gt):#img Tensor[3,h,w], gt Tensor[1,h,w]
      C,H,W=img.shape
      if [H, W]==self.Crop_size:
          return img,gt
      elif H>=self.Crop_size[0] and W>=self.Crop_size[1]:
          return torch.split(T.RandomCrop(self.Crop_size)(torch.cat((img,gt),0)),[C,1],dim=0)#Tensor([3,h,w],[1,h,w])
      else:
          if self.Crop_size[0]/H>self.Crop_size[1]/W:
              Resize=T.Resize([self.Crop_size[0],int(W*self.Crop_size[0]/H)])
          else:
              Resize = T.Resize([int(H*self.Crop_size[1]/W), self.Crop_size[1]])
          return torch.split(T.RandomCrop(self.Crop_size)(Resize(torch.cat((img,gt),0))),[C,1],dim=0)#Tensor([3,h,w],[1,h,w])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    Test=Generate_Dataset('Train/Data','Train/Data',False,256,device)
    test_loader = DataLoader(Test,batch_size=1,shuffle=False)
    for i,traindata in enumerate(test_loader):
        print('i:',i)
        print('data:',traindata['image'])
        Data=traindata['image'].squeeze()
        ImgData=T.ToPILImage()(Data)
        plt.imshow(ImgData)
        plt.show()

        # print('Label:',traindata['label'])
        # Label=traindata['label'].squeeze()
        # ImgLabel=T.ToPILImage()(Label)
        # print(type(ImgLabel))
        # plt.imshow(ImgLabel,cmap='gray')
        # plt.show()#
