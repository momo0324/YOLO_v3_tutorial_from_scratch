from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):#因为同一类别可能会有多个真实检测结果，所以我们使用unique函数来去除重复的元素，即一类只留下一个元素，达到获取任意给定图像中存在的类别的目的
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)#np.unique该函数是去除数组中的重复数字，并进行排序之后输出
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)# new(args, *kwargs) 构建[相同数据类型]的新Tensor
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    #这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    #在特征图上进行多尺度预测, 在GRID每个位置都有三个不同尺度的锚点.predict_transform()利用一个scale得到的feature map预测得到的每个anchor的属性(x,y,w,h,s,s_cls1,s_cls2...),其中x,y,w,h
    #是在网络输入图片坐标系下的值,s是方框含有目标的置信度得分，s_cls1,s_cls_2等是方框所含目标对应每类的概率。输入的feature map(prediction变量) 
    #维度为(batch_size, num_anchors*bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片BxCxHxW存储方式
    #并且将结果的维度变换成(batch_size, grid_size*grid_size*num_anchors, 5+类别数量)的tensor，同时得到每个方框在网络输入图片(416x416)坐标系下的(x,y,w,h)以及方框含有目标的得分以及每个类的得分。
    
    batch_size = prediction.size(0)
    #stride表示的是整个网络的步长，等于图像原始尺寸与yolo层输入的feature mapr尺寸相除，因为输入图像是正方形，所以用高相除即可
    stride =  inp_dim // prediction.size(2)#416//13=32
    # feature map每条边格子的数量，416//32=13
    grid_size = inp_dim // stride
    # 一个方框属性个数，等于5+类别数量
    bbox_attrs = 5 + num_classes
     # anchors数量
    num_anchors = len(anchors)
    # 输入的prediction维度为(batch_size, num_anchors * bbox_attrs, grid_size, grid_size)，类似于一个batch彩色图片
    # 存储方式，将它的维度变换成(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy
    prediction = prediction.transpose(1,2).contiguous()
    # 将prediction维度转换成(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)。不看batch_size，
    # (grid_size*grid_size*num_anchors, bbox_attrs)相当于将所有anchor按行排列，即一行对应一个anchor属性，此时的属性仍然是feature map得到的值
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    # 锚点的维度与net块的height和width属性一致。这些属性描述了输入图像的维度，比feature map的规模大（二者之商即是步幅）。因此，我们必须使用stride分割锚点。变换后的anchors是相对于最终的feature map的尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
   

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    #这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，a，b分别对应这个二维矩阵的x,y坐标的数组，a,b的维度与grid维度一样。每个grid cell的尺寸均为1，故grid范围是[0,12]（假如当前的特征图13*13）
    grid = np.arange(grid_size)
     #x_offset即cx,y_offset即cy，表示当前cell左上角坐标
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)#view是reshape功能，-1表示自适应
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        #这里的x_y_offset对应的是最终的feature map中每个格子的左上角坐标
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    #这里的anchors本来是一个长度为6的list(三个anchors每个2个坐标)，然后在0维上(行)进行了grid_size*grid_size个复制，在1维(列)上
    # 一次复制(没有变化)，即对每个格子都得到三个anchor。Unsqueeze(0)的作用是在数组上添加一维，这里是在第0维上添加的
    # 添加grid_size是为了之后的公式bw=pw×e^tw的tw。
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    #对网络预测得到的矩形框的宽高的偏差值进行指数计算，然后乘以anchors里面对应的宽高(这里的anchors里面的宽高是对应最终的feature map尺寸grid_size)，
    # 得到目标的方框的宽高，这里得到的宽高是相对于在feature map的尺寸
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    # 这里得到每个anchor中每个类别的得分。将网络预测的每个得分用sigmoid()函数计算得到
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    #将相对于最终feature map的方框坐标和尺寸映射回输入网络图片(416x416)，即将方框的坐标乘以网络的stride即可
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
