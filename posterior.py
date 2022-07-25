
from distutils.command.config import config
import numpy as np
import torch
import torch.nn.functional as F
from data_set import KWSDataSet
#from train import generate_batch_data
from network import *
import math

Labels = {"null": 0, "Hello": 1, "xiaogua": 2, "nihao": 3, "xiaoyi": 4, "jixu": 5, "tingzhi": 6, "bofang":7}

class Posterior():
    @staticmethod
    def smooth_frame(outputs, output_length, w_smooth):
        smoothed_outputs = torch.zeros_like(outputs)
        for j in range(outputs.shape[0]):
            h_smooth = max(0, j - w_smooth + 1)
            smoothed_outputs[j] = 1 / (j - h_smooth + 1) * torch.sum(outputs[h_smooth:j + 1], dim=0)
        return smoothed_outputs
    
    @staticmethod
    def get_max_list(outputs, output_length, w_max):
        max_list = []
        frame, label = outputs.shape
        for j in range(frame):
            h_max = max(0, j - w_max + 1)
            max_value = torch.max(outputs[h_max:j + 1], dim=0).values[:].unsqueeze(0)
            max_list.append(torch.squeeze(max_value, dim=0))
        return max_list
    
    @staticmethod 
    def get_confidence(outputs, w_smooth, w_max):
        confidence = [0.0]*4
        outputs = Posterior.smooth_frame(outputs, len(outputs[0]), w_smooth)
        outputs = Posterior.get_max_list(outputs, len(outputs[0]), w_max)
        
        for index, item in enumerate(outputs):
            if(math.sqrt(item[1]*item[2]) > confidence[0]):
                confidence[0] = math.sqrt(item[1]*item[2])
            if(math.sqrt(item[3]*item[4]) > confidence[1]):
                confidence[1] = math.sqrt(item[3]*item[4])
            if(math.sqrt(item[5]*item[7]) > confidence[2]):
                confidence[2] = math.sqrt(item[5]*item[7])
            if(math.sqrt(item[6]*item[7]) > confidence[3]):
                confidence[3] = math.sqrt(item[6]*item[7])
        return confidence
    @staticmethod 
    def get_video_real_label(video_labels):
        for i in range(video_labels.shape[0]):
            if(video_labels[i] == Labels["Hello"]):
                return 1
            if(video_labels[i] == Labels["nihao"]):
                return 2
            if(video_labels[i] == Labels["jixu"]):
                return 3
            if(video_labels[i] == Labels["tingzhi"]): 
                return 4
        return 0
    
    
def generate_batch_data(batch):
    batch_X = []
    batch_Y = []
    for index, data in enumerate(batch):
        batch_datas = []
        batch_labels = []
        frame_list_2d = torch.split(data, 1, dim=0)
        for frame_item in frame_list_2d:
            batch_item_list = torch.split(frame_item, 40, dim=1)
            batch_datas.append(batch_item_list[0])
            batch_labels.append(batch_item_list[1])
        temp_x, temp_y = combine_frames_as_frame(batch_datas, batch_labels)
        if index==0:
            batch_X = temp_x
            batch_Y = temp_y
        else: 
            batch_X = torch.cat((batch_X,temp_x),0)
            batch_Y = torch.cat((batch_Y,temp_y),0)
    return batch_X, batch_Y


def combine_frames_as_frame(datas, labels):
    re_datas = torch.zeros(1, 41, 40)
    re_labels = torch.zeros(1)
    for index,data in enumerate(datas):
        if(index >= 30 and index < len(datas)-10):
            sum_tensor = datas[index-30]
            for j in range(index-29,index+11):
                sum_tensor = torch.cat((sum_tensor,datas[j]), 0)
            
            re_datas = torch.cat((re_datas, torch.unsqueeze(sum_tensor, 0)), 0)
            re_labels = torch.cat((re_labels, torch.squeeze(labels[index], 0)), 0)
    return re_datas, re_labels

def test_posterior():
    test_dataset_path = "/home/disk1/user5/wyz/DataSet/TestSet"
    test_set = KWSDataSet(test_dataset_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=generate_batch_data,
                                                shuffle=True, num_workers=16, pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = KWSNet().to(device)
    net.load_state_dict(torch.load("/home/disk1/user5/deep_kws-3.pth"))

    for index, data in enumerate(test_loader):
        confidences = []
        inputs = data[0].to(device)
        labels =data[1].to(device, dtype=torch.int64)
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        
        confidence = Posterior.get_confidence(outputs,30,100)
        print(outputs)
        print(labels)
        #print(confidence)
        #print(Posterior.get_video_real_label(labels))
        #confidences.append(confidence)

