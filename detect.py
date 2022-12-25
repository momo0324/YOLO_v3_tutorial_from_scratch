from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    #检测模块的参数转换
    """
    Parse arguements to the detect module
    
    """
    #创建一个ArgumentParser对象，格式: 参数名, 目标参数(dest是字典的key),帮助信息,默认值,类型 
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
                        # images是所有测试图片所在的文件夹
    parser.add_argument("--det", dest = 'det', help =  #det保存检测结果的目录
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
                        #reso输入图像的分辨率，可用于在速度与准确度之间的权衡
    
    return parser.parse_args()# 返回转换好的结果
    
args = arg_parse()# args是一个namespace类型的变量，即argparse.Namespace, 可以像easydict一样使用,就像一个字典，key来索引变量的值
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()# 检测GPU环境是否可用



num_classes = 80# 表示coco数据集有80类
classes = load_classes("data/coco.names")
#将类别文件载入到我们的程序中，coco.names文件中保存的是所有类别的名字，load_classes()返回一个列表classes，每个元素是一个类别的名字




#Set up the neural network
#初始化网络并载入权重
print("Loading network.....")# Darknet类中初始化时得到了网络结构和网络的参数信息，保存在net_info，module_list中
model = Darknet(args.cfgfile)# 将权重文件载入，并复制给对应的网络结构model中
model.load_weights(args.weightsfile)
print("Network successfully loaded")
#网络输入数据大小
model.net_info["height"] = args.reso
# model类中net_info是一个字典。’’height’’是图片的宽高，因为图片缩放到416x416，所以宽高一样大
inp_dim = int(model.net_info["height"])#inp_dim是网络输入图片尺寸
assert inp_dim % 32 == 0 # 如果设定的输入图片的尺寸不是32的位数或者不大于32，抛出异常
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()
#变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的

read_dir = time.time()
#read_dir 是一个用于测量时间的检查点,开始计时
#Detection phase
try:
     #从磁盘读取图像或从目录读取多张图像。图像的路径存储在一个名为 imlist 的列表中,imlist列表保存了images文件中所有图片的完整路径，一张图片路径对应一个元素 
     #osp.realpath('.')得到了图片所在文件夹的绝对路径，images是测试图片文件夹，listdir(images)得到了images文件夹下面所有图片的名字
     #通过join()把目录（文件夹）的绝对路径和图片名结合起来，就得到了一张图片的完整路径
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:# 如果上面的路径有错，只得到images文件夹绝对路径即可
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()#存储结果目录
    
if not os.path.exists(args.det):#如果保存检测结果的目录（由 det 标签定义）不存在，就创建一个
    os.makedirs(args.det)

load_batch = time.time()
# 开始载入图片的时间 
#load_batch - read_dir 得到读取所有图片路径的时间
loaded_ims = [cv2.imread(x) for x in imlist]
#使用 OpenCV 来加载图像，将所有的图片读入，一张图片的数组在loaded_ims列表中保存为一个元素

# 加载全部待检测图像
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
#除了转换后的图像，我们也会维护一个列表im_dim_list用于保存原始图片的维度。一个元素对应一张图片的宽高,opencv读入的图片矩阵对应的是 HxWxC
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
#repeat(*size), 沿着指定维度复制数据，size维度必须和数据本身维度要一致


leftover = 0#创建 batch，将所有测试图片按照batch_size分成多个batch
if (len(im_dim_list) % batch_size):
    leftover = 1# 如果测试图片的数量不能被batch_size整除，leftover=1
    #如果batch size 不等于1，则将一个batch的图片作为一个元素保存在im_batches中，按照if语句里面的公式计算。如果batch_size=1,则每一张图片作为一个元素保存在im_batches中

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    # 前面的im_batches变量将所有的图片以BxCxHxW的格式保存。而这里将一个batch的所有图片在B这个维度(第0维度)上进行连接，torch.cat()默认在0维上进行连接。将这个连接后的tensor作为im_batches列表的一个元素。
    #第i个batch在前面的im_batches变量中所对应的元素就是i*batch_size: (i + 1)*batch_size，但是最后一个batch如果用(i + 1)*batch_size可能会超过图片数量的len(im_batches)长度，所以取min((i + 1)*batch_size, len(im_batches)             
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0


if CUDA:# 开始计时，计算开始检测的时间。start_det_loop - load_batch 为读入所有图片并将它们分成不同batch的时间 
    im_dim_list = im_dim_list.cuda()
    
start_det_loop = time.time()
# enumerate返回im_batches列表中每个batch在0维连接成一个元素的tensor和这个tensor在im_batches中的序号。本例子中batch只有一张图片
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
        # 取消梯度计算
    with torch.no_grad():
      # Variable(batch)将图片生成一个可导tensor
      # prediction是一个batch所有图片通过yolov3模型得到的预测值，维度为1x10647x85，三个scale的图片每个scale的特征图大小为13x13,26x26,52x52,一个元素看作一个格子，每个格子有3个anchor，将一个anchor保存为一行，
      #所以prediction一共有(13x13+26x26+52x52)x3=10647行，一个anchor预测(x,y,w,h,s,s_cls1,s_cls2...s_cls_80)，一共有85个元素。所以prediction的维度为Bx10647x85，加为这里batch_size为1，所以prediction的维度为1x10647x85  
        prediction = model(Variable(batch), CUDA)
      # 结果过滤.这里返回了经过NMS后剩下的方框，最终每个方框的属性为(ind,x1,y1,x2,y2,s,s_cls,index_cls) ind是这个方框所属图片在这个batch中的序号，x1,y1是在网络输入图片(416x416)坐标系中，方框左上角的坐标；x2,y2是方框右下角的坐标。
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()
    # 如果从write_results()返回的一个batch的结果是一个int(0)，表示没有检测到时目标，此时用continue跳过本次循环

    if type(prediction) == int:
        # 在imlist中，遍历一个batch所有的图片对应的元素(即每张图片的存储位置和名字)，同时返回这张图片在这个batch中的序号im_num

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            # 计算图片在imlist中所对应的序号,即在所有图片中的序号
            im_id = i*batch_size + im_num
            # 打印图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            # 输出本次处理图片所有检测到的目标的名字
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue
        # prediction[:,0]取出了每个方框在所在图片在这个batch(第i个batch)中的序号，加上i*batch_size，就将prediction中每个框(一行)的第一个元素（维度0）变成了这个框所在图片在imlist中的序号，即在所有图片中的序号
    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction 
          # output将每个batch的输出结果在0维进行连接，即在行维度上连接，每行表示一个检测方框的预测值。最终本例子中的11张图片检测得到的结果output维度为 34 x 8
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
     # objs列表包含了本次处理图片中所有检测得到的方框所包含目标的类别名称。每个元素对应一个检测得到的方框所包含目标的类别名称。for x in output遍历output中的每一行(即一个方框的预测值)得到x，如果这个方
     #框所在图片在所有图片中的序号等于本次处理图片的序号，则用classes[int(x[-1])找到这个方框包含目标类别在classes中对应的类的名字
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        # classes在之前的语句classes = load_classes("data/coco.names")中就是为了把类的序号转为字符名字
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        # 打印本次处理图片运行的时间，用一个batch的平均运行时间来表示。.3f就表示保留三位小数点的浮点
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))# 输出本次处理图片所有检测到的目标的类别名字
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
         # 保证gpu和cpu同步，否则，一旦 GPU 工作排队了并且 GPU 工作还远未完成，那么 CUDA 核就将控制返回给 CPU      
try:
    output
except NameError:
    print ("No detections were made")
    exit()# 当所有图片都有没检测到目标时，退出程序
    # 最后输出output_recast - start_det_loop计算的是从开始检测，到去掉低分，NMS操作的时间

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)
# 将相对于输入网络图片(416x416)的方框属性变换成原图按照纵横比不变进行缩放后的区域的坐标。
#scaling_factor*img_w和scaling_factor*img_h是图片按照纵横比不变进行缩放后的图片，即原图是768x576按照纵横比长边不变缩放到了416*372。
#经坐标换算,得到的坐标还是在输入网络的图片(416x416)坐标系下的绝对坐标，但是此时已经是相对于416*372这个区域的坐标了，而不再相对于(0,0)原点。


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2


# 将方框坐标(x1,y1,x2,y2)映射到原始图片尺寸上，直接除以缩放系数即可。output[:,1:5]维度为34x4，scaling_factor维度是34x1.相除时会利用广播性质将scaling_factor扩展为34x4的tensor
output[:,1:5] /= scaling_factor

# 如果映射回原始图片中的坐标超过了原始图片的区域，则x1,x2限定在[0,img_w]内，img_w为原始图片的宽度。如果x1,x2小于0.0，令x1,x2为0.0，如果x1,x2大于原始图片宽度，令x1,x2大小为图片的宽度。
#同理，y1,y2限定在0,img_h]内，img_h为原始图片的高度。clamp()函数就是将第一个输入对数的值限定在后面两个数字的区间
for i in range(output.shape[0]):
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()# 开始载入颜色文件的时间
 # 绘图
class_load = time.time()# 开始画方框的文字的时间
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    # x为映射到原始图片中一个方框的属性(ind,x1,y1,x2,y2,s,s_cls,index_cls)，results列表保存了所有测试图片，一个元素对应一张图片
    c1 = tuple(x[1:3].int())# c1为方框左上角坐标x1,y1
    c2 = tuple(x[3:5].int())# c2为方框右下角坐标x2,y2
    img = results[int(x[0])]# 在results中找到x方框所对应的图片，x[0]为方框所在图片在所有测试图片中的序号
    cls = int(x[-1])
    color = random.choice(colors)# 随机选择一个颜色，用于后面画方框的颜色
    label = "{0}".format(classes[cls])# label为这个框所含目标类别名字的字符串
    cv2.rectangle(img, c1, c2,color, 1)# 在图片上画出(x1,y1,x2,y2)矩形，即我们检测到的目标方框
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]# 得到一个包含目标名字字符的方框的宽高
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4# 得到包含目标名字的方框右下角坐标c2，这里在x,y方向上分别加了3、4个像素
    cv2.rectangle(img, c1, c2,color, -1)# 在图片上画一个实心方框，我们将在方框内放置目标类别名字
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

# 开始逐条绘制output中结果.将每个框在对应图片上画出来，同时写上方框所含目标名字。map函数将output传递给map()中参数是函数的那个参数，每次传递一行。
#而lambda中x就是output中的一行，维度为1x8。loaded_ims列表保存了所有图片内容数组,一个元素对应一张图片，原地修改了loaded_ims 之中的图像，使之还包含了目标类别名字。
list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))


end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
# 从开始检测到到去掉低分，NMS操作得到output的时间. 
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
#这里output映射回原图的时间
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))# 画框和文字的时间
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
    
