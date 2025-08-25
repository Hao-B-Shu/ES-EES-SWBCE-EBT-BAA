import os
import random
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from scipy import io
import torch.nn.functional as F

normalize=T.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
totensor=T.ToTensor()

def Save_label_to_mat(Label_dir,Save_dir): # Nonzero value will set be 1
    Label_name=os.listdir(Label_dir)
    Label_name.sort()
    for index in range (len(Label_name)):
        Label_file=os.path.join(Label_dir, Label_name[index])
        Label =cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)
        Label[Label!=0]=1
        io.savemat(Save_dir+'/'+Label_name[index][:-4]+'.mat',{'groundTruth':[{'Boundaries':Label}]})

def Add_label_to_data(Label_dir,Data_dir):
    Label_name=os.listdir(Label_dir)
    Label_name.sort()
    for index in range (len(Label_name)):
        Label_file=os.path.join(Label_dir, Label_name[index])
        img=cv2.imread(Label_file)
        Label =cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)
        Label[Label!=0]=255
        img[img != 0] = 255
        cv2.imwrite(os.path.join(Label_dir, 'Label'+Label_name[index][:-4] + '.png'), Label.astype(np.uint8))
        cv2.imwrite(os.path.join(Data_dir, 'Label' + Label_name[index][:-4] + '.jpg'), img.astype(np.uint8))


##################################Preprocess################################################

def Random_split_data(Data_dir,Label_dir,Save_data_dir,Save_label_dir,Save_data_dir_1,Save_label_dir_1,split_to,Save_data_dir_2=None,Save_label_dir_2=None):

    Data_name = sorted(os.listdir(Data_dir))
    Label_name = sorted(os.listdir(Label_dir))
    assert len(Data_name)==len(Label_name)
    assert len(split_to)==2 or 3
    if len(split_to)==2:
        assert split_to[0] + split_to[1] <= len(Data_name)
    if len(split_to)==3:
        assert split_to[0] + split_to[1]+ split_to[2] <= len(Data_name)

    Pair = []
    for index in range (len(Data_name)):
        Pair.append((Data_name[index],Label_name[index]))

    random.shuffle(Pair)

    for index in range (split_to[0]):
        data_file=os.path.join(Data_dir, Pair[index][0])
        label_file = os.path.join(Label_dir, Pair[index][1])
        data = cv2.imread(data_file)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        if Save_data_dir!=None:
            data_save_path = os.path.join(Save_data_dir, Pair[index][0])
            cv2.imwrite(data_save_path, data)
        if Save_label_dir!=None:
            label_save_path = os.path.join(Save_label_dir, Pair[index][1])
            cv2.imwrite(label_save_path, label)

    for index in range (split_to[0],split_to[0]+split_to[1]):
        data_file = os.path.join(Data_dir, Pair[index][0])
        label_file = os.path.join(Label_dir, Pair[index][1])
        data = cv2.imread(data_file)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        if Save_data_dir_1 != None:
            data_save_path = os.path.join(Save_data_dir_1, Pair[index][0])
            cv2.imwrite(data_save_path, data)
        if Save_label_dir_1 != None:
            label_save_path = os.path.join(Save_label_dir_1, Pair[index][1])
            cv2.imwrite(label_save_path, label)

    if len(split_to)==3:
        for index in range(split_to[0] + split_to[1],split_to[0] + split_to[1]+ split_to[2]):
            data_file = os.path.join(Data_dir, Pair[index][0])
            label_file = os.path.join(Label_dir, Pair[index][1])
            data = cv2.imread(data_file)
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            if Save_data_dir_2 != None:
                data_save_path = os.path.join(Save_data_dir_2, Pair[index][0])
                cv2.imwrite(data_save_path, data)
            if Save_label_dir_2 != None:
                label_save_path = os.path.join(Save_label_dir_2, Pair[index][1])
                cv2.imwrite(label_save_path, label)

def Check_name(Data_dir,Edge_dir):
    Data_name = os.listdir(Data_dir)
    Edge_name = os.listdir(Edge_dir)
    Data_name.sort()
    Edge_name.sort()
    assert len(Data_name)==len(Edge_name)
    for index in range (len(Data_name)):
        assert Data_name[index][0:-4]==Edge_name[index][0:-4]
    print('Check passed')
    print('Number',len(Data_name))

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

def Half_w(image,label):
    H,W,_=image.shape
    new_w=int(W/2)
    Img1=image[:,0:new_w,:]
    Img2 = image[:, new_w:2*new_w, :]
    Lab1 = label[:, 0:new_w]
    Lab2 = label[:, new_w:2 * new_w]
    return Img1,Img2,Lab1,Lab2

def Half_h(image,label):
    H,W,_=image.shape
    new_h=int(H/2)
    Img1=image[0:new_h,:,:]
    Img2 = image[ new_h:2*new_h,:, :]
    Lab1 = label[0:new_h,: ]
    Lab2 = label[ new_h:2 * new_h,:]
    return Img1,Img2,Lab1,Lab2

def Data_aug_half(img_dir,lab_dir,thr=320):
    img_name = sorted(os.listdir(img_dir))
    lab_name = sorted(os.listdir(lab_dir))

    Split_list_img=[]
    Split_list_lab = []

    for index in range (len(img_name)):
        Image_file = os.path.join(img_dir, img_name[index])
        Label_file = os.path.join(lab_dir, lab_name[index])
        img = cv2.imread(Image_file, cv2.IMREAD_COLOR)
        lab = cv2.imread(Label_file, cv2.IMREAD_GRAYSCALE)

        list_img=[img]
        list_lab=[lab]
        out_img=[]
        out_lab=[]
        while len(list_img)!=0:
            tmp_img=list_img[0]
            tmp_lab=list_lab[0]
            h,w=tmp_lab.shape
            if w>=2*thr:
                img1,img2,lab1,lab2=Half_w(tmp_img,tmp_lab)
                list_img.append(img1)
                list_img.append(img2)
                list_lab.append(lab1)
                list_lab.append(lab2)
                list_img.remove(list_img[0])
                list_lab.remove(list_lab[0])
            elif  h>=2*thr:
                img1,img2,lab1,lab2=Half_h(tmp_img,tmp_lab)
                list_img.append(img1)
                list_img.append(img2)
                list_lab.append(lab1)
                list_lab.append(lab2)
                list_img.remove(list_img[0])
                list_lab.remove(list_lab[0])
            else:
                out_img.append(tmp_img)
                out_lab.append(tmp_lab)
                list_img.remove(list_img[0])
                list_lab.remove(list_lab[0])
        for i in range(len(out_img)):
            Split_list_img.append([img_name[index][:-4] + '_'+str(i+1), out_img[i]])
            Split_list_lab.append([lab_name[index][:-4] + '_'+str(i+1), out_lab[i]])

    return Split_list_img,Split_list_lab

def Filp(image, label,code):
    image = cv2.flip(image, code)
    label = cv2.flip(label, code)
    return image, label

def Gamma_correction(img,lab,gamma=1):
    corrected_img = (Unit_Normalize(img)/255)**gamma
    corrected_img = (corrected_img*255).astype(np.uint8)
    return corrected_img,lab

def Rotation_and_crop(image, label,angle,crop=None):
    h, w = label.shape
    Rotation_Matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, Rotation_Matrix, (w, h))
    label = cv2.warpAffine(label, Rotation_Matrix, (w, h))

    if crop:
        angle_crop=angle%180
        if angle_crop>90:
            angle_crop=180-angle_crop
        theta=angle_crop*np.pi/180
        hw_ratio=h/w
        tan_theta=np.tan(theta)
        numerator=np.cos(theta)+np.sin(theta)*tan_theta

        if h>w:
            r=hw_ratio
        else:
            r=1/hw_ratio
        denominator=r*tan_theta+1

        crop_mult=numerator/denominator
        w_crop=int(round(crop_mult*w))
        h_crop=int(round(crop_mult*h))
        x0=int((w-w_crop)/2)
        y0=int((h-h_crop)/2)
        image=image[y0:h_crop,x0:w_crop,:]
        label = label[y0:h_crop,x0:w_crop]
    return image, label

def Rotation_90(image, label,angle):
    assert angle%90==0
    h, w = label.shape
    if angle%180==0:
        Rotation_Matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, Rotation_Matrix, (w, h))
        label = cv2.warpAffine(label, Rotation_Matrix, (w, h))
    else:
        if h>w:
            h_pad = 0
            w_pad=int((h-w)/2)
        elif h<w:
            h_pad = int((w - h) / 2)
            w_pad = 0
        else:
            h_pad=0
            w_pad=0
        Img_pad = cv2.copyMakeBorder(image, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        Lab_pad = cv2.copyMakeBorder(label, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        Rotation_Matrix = cv2.getRotationMatrix2D((w // 2+w_pad, h // 2+h_pad), angle, 1.0)
        image_pad = cv2.warpAffine(Img_pad, Rotation_Matrix, (w+2*w_pad, h+2*h_pad))
        label_pad = cv2.warpAffine(Lab_pad, Rotation_Matrix, (w+2*w_pad, h+2*h_pad))
        image=image_pad[w_pad:w_pad+w,h_pad:h_pad+h,:]
        label = label_pad[w_pad:w_pad + w, h_pad:h_pad + h]

    return image, label

def Data_aug_RFG(img_dir,lab_dir,imgaug_dir,labaug_dir,angle_num=4,flip=True,gamma=[1],crop=True,Only_90=True,Split_thr=320):

    img_list,lab_list=Data_aug_half(img_dir,lab_dir,Split_thr)

    for index in range (len(img_list)):

        [img_name,img] = img_list[index]
        [lab_name,lab] = lab_list[index]

        for value in range(len(gamma)):
            if Only_90==True:
                angle_num=4
                for i in range (angle_num):
                    angle=i*360/angle_num
                    Img, Lab = Rotation_90(img, lab, angle)
                    Img,Lab=Gamma_correction(img=Img,lab=Lab,gamma=gamma[value])
                    cv2.imwrite(imgaug_dir + '/' + img_name + '_R'+str(angle)+ '_G'+str(gamma[value])+'.jpg', Img)
                    cv2.imwrite(labaug_dir + '/' + lab_name + '_R'+str(angle)+ '_G'+str(gamma[value])+'.png', Lab)
                    if flip:
                        Img, Lab = Filp(Img, Lab, 1)
                        cv2.imwrite(imgaug_dir + '/' + img_name + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.jpg', Img)
                        cv2.imwrite(labaug_dir + '/' + lab_name + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.png', Lab)
            else:
                for i in range (angle_num):
                    angle=i*360/angle_num
                    Img, Lab = Rotation_and_crop(img, lab, angle, crop=crop)
                    Img,Lab=Gamma_correction(img=Img,lab=Lab,gamma=gamma[value])
                    cv2.imwrite(imgaug_dir + '/' + img_name + '_R'+str(angle)+ '_G'+str(gamma[value])+'.jpg', Img)
                    cv2.imwrite(labaug_dir + '/' + lab_name + '_R'+str(angle)+ '_G'+str(gamma[value])+'.png', Lab)
                    if flip:
                        Img, Lab = Filp(Img, Lab, 1)
                        cv2.imwrite(imgaug_dir + '/' + img_name + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.jpg', Img)
                        cv2.imwrite(labaug_dir + '/' + lab_name + '_R'+str(angle)+ '_G'+str(gamma[value]) + 'F.png', Lab)


if __name__=='__main__':

############## The codes are for data preprocessing ###############

    print('Processing')

    # ###########################Random partition of a dataset###########################
    # Dir_Proj='' # Project dir
    # split_to = [400, 100] # [Train data, Test data] or [Train data, Test data, Val data]
    #
    # Dataset_Data='' # Path of the dataset (image)
    # Dataset_Label=''# Path of the dataset (label)
    # Train_Data=Dir_Proj+'Train_Data'
    # Train_Label=Dir_Proj+'Train_Label'
    # Test_Data=Dir_Proj+'Test_Data'
    # Test_Label=Dir_Proj+'Test_Label'
    # Random_split_data(Data_dir=Dataset_Data, Label_dir=Dataset_Label, Save_data_dir=Train_Data,Save_label_dir=Train_Label,
    #                   Save_data_dir_1=Test_Data, Save_label_dir_1=Test_Label,Save_data_dir_2=None, Save_label_dir_2=None, split_to=split_to)
    # Check_name(Train_Data, Train_Label)
    #
    # ##################Data augmentation: set the path below and run the code, all dirs should be built manually###################
    # Data_dir=Train_Data
    # Label_dir=Train_Label
    # Dataaug_dir=Dir_Proj+'Train_Data_Bigaug' # The path of dir for saving training data after augmentations
    # Labelaug_dir=Dir_Proj+'Train_Label_Bigaug'# The path of dir for saving training labels after augmentations
    # Data_aug_RFG(img_dir=Data_dir,lab_dir=Label_dir,imgaug_dir=Dataaug_dir,labaug_dir=Labelaug_dir,angle_num=4,flip=True,gamma=[1],crop=True,Only_90=True,Split_thr=320)
    # # Rotations flips and crops, when Only_90=True, angle_num will be reset to be 4 and crop will be reset to be False
    # Add_label_to_data(Label_dir=Labelaug_dir,Data_dir=Dataaug_dir)
    # Check_name(Dataaug_dir,Labelaug_dir)
    # ########################################################################################
    #
    # #############################Use the following to save .png image to .mat type##############
    # Save_label_to_mat(Label_dir=Test_Label, Save_dir=Test_Label + '_mat')
    # ############################################################################################
