import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from train import generate_batch_data
#from train import combine_frames_as_frame
from posterior import Posterior
from network import KWSNet
from data_set import KWSDataSet

# thresholds当前枚举的置信度阈值
# total_count：测试集数据数 false_count：测试集负样例数
# eval_res[0:false_count] 所有负样例
#   在这些负样例中，eval_res[i]['score']存放第i个音频中所有识别语句置信度最高的值，按置信度从小到大排序
# eval_res[false_count:] 所有正样例
#   在这些正样例中，eval_res[i]['score']存放第i个音频中正确识别语句置信度最高的值，按置信度从小到大排序
# false_ind：当前枚举的thresholds在负样例（eval_res[0:false_count][score]）中的位置
# true同理
# (false_count - false_ind)表示在当前thresholds下，误唤醒的数量
# (true_ind - false_count)表示在当前thresholds下，误拒绝的数量
thresholds = np.arange(0, 1.0, 0.001)


test_dataset_path = "/home/disk1/user5/wyz/DataSet/TestSet"
NET_MODEL = "/home/disk1/user5/wyz/Models/deep_kws-data2-net1-9.pth"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

def get_model_roc(test_dataset_url, model_url, device):
    
    test_set = KWSDataSet(test_dataset_url)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, collate_fn=generate_batch_data,
                                                shuffle=True, num_workers=16, pin_memory=True)
    net = KWSNet().to(device)
    net.load_state_dict(torch.load(model_url))
    count = [0]*5
    total_counts = []
    false_counts = []
    eval_res = [[] for _ in range(4)]
    fal_res = [[] for _ in range(4)]
    tru_res = [[] for _ in range(4)]
    for index, data in enumerate(test_loader):
        
        
        labels =data[1].to(device, dtype=torch.int64)
        rel_sentence_label = Posterior.get_video_real_label(labels)
        
        inputs = data[0].to(device)
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        confidence = Posterior.get_confidence(outputs,30,100)
        
        if rel_sentence_label==0:
            count[4] += 1
            for i in range(4):
                fal_res[i].append({'score': confidence[i]})
        else:
            for i in range(4):
                if i == (rel_sentence_label-1):
                    count[i] += 1
                    tru_res[i].append({'score': confidence[i]})
                    
        if index % 100 == 99:
            print(index+1)
    for i in range(4):
        fal_res[i].sort(key=lambda x: x['score'])
        tru_res[i].sort(key=lambda x: x['score'])
        eval_res[i] = fal_res[i]+tru_res[i]
    total_counts.append(count[4]+count[0])
    total_counts.append(count[4]+count[1])
    total_counts.append(count[4]+count[2])
    total_counts.append(count[4]+count[3])
    false_counts = [count[4]]*4
    roc_draw(total_counts, false_counts, eval_res)

    
    
    
    

def roc_draw(total_count, false_count, eval_res):
    for i in range(len(total_count)):
        roc_one_line(total_count[i], false_count[i], eval_res[i])
    plt.xlabel('False Alarms')
    plt.ylabel('False Rejects')
    plt.title('FA VS FR')
    plt.xlim(0,1.0)
    plt.ylim(0,1.0)
    plt.savefig('ROC4.png', dpi=300)
    plt.clf()
    
    
def roc_one_line(total_count, false_count, eval_res):
    false_ind = 0
    true_ind = false_count
    true_count = total_count - false_count
    fas = []
    frs = []
    best_add_thres = float('inf')
    best_add_score = float('inf')
    for threshold in thresholds:
        while false_ind < false_count and eval_res[false_ind]['score'] < threshold:
            false_ind += 1
        while true_ind < total_count and eval_res[true_ind]['score'] < threshold:
            true_ind += 1

        # FA = FP / (TN + FP)
        fa = (false_count - false_ind) / false_count
        # FR = FN / (TP + FN)
        fr = (true_ind - false_count) / true_count

        fas.append(fa)
        frs.append(fr)

    # draw plots showing the result
    plt.plot(fas, frs)

    
get_model_roc(test_dataset_path, NET_MODEL, device)