
from random import shuffle
from network import *
import torch
import torch.nn as nn
import torch.optim as optim
from data_set import KWSDataSet

Labels = {"null": 0, "Hello": 1, "xiaogua": 2, "nihao": 3, "xiaoyi": 4, "jixu": 5, "tingzhi": 6, "bofang":7}
key_class = ("Hello", "xiaogua", "nihao", "xiaoyi", "jixu", "tingzhi", "bofang", "null")
train_dataset_path = "/home/disk1/user5/wyz/DataSet/TrainSet"
valid_dataset_path = "/home/disk1/user5/wyz/DataSet/ValidSet"
test_dataset_path = "/home/disk1/user5/wyz/DataSet/TestSet"

def generate_batch_data(batch):
    batch_datas = []
    batch_labels = []
    batch = torch.squeeze(batch, dim=0)
    frame_list_2d = torch.split(batch, 1, dim=0)

    frame_list_1d = []
    for frame_2d in frame_list_2d:
        frame_list_1d.append(frame_2d)

    for frame_item in frame_list_1d:
        batch_item_list = torch.split(frame_item, 40, dim=1)
        batch_datas.append(batch_item_list[0])
        batch_labels.append(batch_item_list[1])
    return combine_frames_as_frame(batch_datas, batch_labels)


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


def split_batchs(batch_size, datas, labels):
    data_batch_list =  list(torch.split(datas, batch_size, dim=0))
    label_batch_list = list(torch.split(labels, batch_size, dim=0))
    data_batch_list.pop()
    label_batch_list.pop()
    return data_batch_list, label_batch_list
    

fbanks_batch_size = 1
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

'''
train_set = KWSDataSet(train_dataset_path)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=fbanks_batch_size,
                                            shuffle=True, num_workers=16, pin_memory=True)
'''

'''
valid_set = KWSDataSet(valid_dataset_path)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=fbanks_batch_size,
                                            shuffle=True, num_workers=16, pin_memory=True)
'''

net = KWSNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.60)


'''
inputs_list = []
labels_list = []
for index, data in enumerate(train_loader):
    long_inputs, long_labels = generate_batch_data(data)
    unit_audio_inputs, unit_audio_labels = split_batchs(16, long_inputs, long_labels)
    inputs_list = inputs_list + unit_audio_inputs
    labels_list = labels_list + unit_audio_labels
print("compelet proccess of train frame")

torch.save(inputs_list, "./inputs_list_1.pth")
torch.save(labels_list, "./labels_list_1.pth")

'''

'''
valid_inputs_list = []
valid_labels_list = []
for index, data in enumerate(valid_loader):
    long_inputs, long_labels = generate_batch_data(data)
    unit_audio_inputs, unit_audio_labels = split_batchs(16, long_inputs, long_labels)
    valid_inputs_list = valid_inputs_list + unit_audio_inputs
    valid_labels_list = valid_labels_list + unit_audio_labels
print("compelet proccess of valid frame")

torch.save(valid_inputs_list, "./valid_inputs_list_1.pth")
torch.save(valid_labels_list, "./valid_labels_list_1.pth")

'''

inputs_list = torch.load("wyz/DataSet/inputs_list_1.pth")
labels_list = torch.load("wyz/DataSet/labels_list_1.pth")

valid_inputs_list = torch.load("wyz/DataSet/valid_inputs_list_1.pth")
valid_labels_list = torch.load("wyz/DataSet/valid_labels_list_1.pth")

print("compelet proccess of frame")

for epoch in range(3): 
    print("learning rate of epoch-%d: %f" % (epoch, optimizer.param_groups[0]['lr']))
    count_epoch = 0
    loss_epoch_sum = 0.0
        
    for index, inputs in enumerate(inputs_list):
            
        inputs = inputs.to(device)
        labels = labels_list[index].to(device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
            
        loss_epoch_sum = loss_epoch_sum + loss
        #count_epoch = count_epoch+1
            
        if index % 2000000 == 1999999:
            print(f'[{epoch + 1}, {index + 1:5d}] loss: {loss_epoch_sum / (index + 1):.7f}')
            StepLR.step()
            print("learning rate of epoch-%d: %f" % (epoch, optimizer.param_groups[0]['lr']))
                
    torch.save(net.state_dict(), 'deep_kws-'+str(epoch)+'.pth')
    print(f'[{epoch + 1}] average train loss: {loss_epoch_sum / (index + 1):.7f}')
    
    ###################valid####################
    count_valid = 0
    loss_sum_valid = 0.0
    
    for index, frame in enumerate(valid_inputs_list):
            
        frame = frame.to(device)
        labels = valid_labels_list[index].to(device, dtype=torch.int64)
        outputs = net(frame)
        
        loss = criterion(outputs, labels)
        
        loss_sum_valid = loss_sum_valid + loss
    
    print(f'[{epoch + 1}] average valid loss: {loss_sum_valid / (index+1):.7f}')
    

torch.save(net.state_dict(), 'deep_kws.pth')
print('Finished Training')
