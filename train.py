from calendar import c
from random import shuffle
from network import KWSNet
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
        frame_list_1d.append(torch.squeeze(frame_2d,dim=0))

    for frame_item in frame_list_1d:
        batch_item_list = torch.split(frame_item, 40, dim=0)
        batch_datas.append(batch_item_list[0])
        batch_labels.append(batch_item_list[1])
    return combine_frames_as_frame(batch_datas, batch_labels)


def combine_frames_as_frame(datas, labels):
    re_datas = []
    re_labels = []
    for index,data in enumerate(datas):
        if(index >= 30 and index < len(datas)-10):
            sum_tensor = data
            for j in range(index-30,index+10):
                sum_tensor = torch.cat((sum_tensor,datas[j]), 0)
            re_datas.append(sum_tensor)
            re_labels.append(labels[index])
    return re_datas, re_labels



fbanks_batch_size = 1
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

train_set = KWSDataSet(train_dataset_path)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=fbanks_batch_size,
                                            shuffle=True, num_workers=16)

valid_set = KWSDataSet(valid_dataset_path)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=fbanks_batch_size,
                                            shuffle=True, num_workers=16)

net = KWSNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0000005, momentum=0.9)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.60)

count = 0
loss_sum = 0.0
for epoch in range(10): 
    print("learning rate of epoch-%d: %f" % (epoch, optimizer.param_groups[0]['lr']))
    count_epoch = 0
    loss_epoch_sum = 0.0
    
    for index,data in enumerate(train_loader):
        data_list, label_list = generate_batch_data(data)
        data_list.to(device)
        for i, item in enumerate(data_list):
            inputs, labels = data_list[i], label_list[i].to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(torch.unsqueeze(outputs,0), labels)
            loss_epoch_sum = loss_epoch_sum + loss
            count_epoch = count_epoch+1
            loss.backward()
            optimizer.step()
            if count_epoch % 2000 == 1999:
                print(f'[{epoch + 1}, {count_epoch + 1:5d}] loss: {loss_epoch_sum / count_epoch:.7f}')
    loss_sum = loss_sum + loss_epoch_sum
    count = count + count_epoch
    StepLR.step()
    
    count_valid = 0
    loss_sum_valid = 0
    for index, data in enumerate(valid_loader):
        data_list, label_list = generate_batch_data(data)
        for i, item in enumerate(data_list):
            inputs, labels = data_list[i].to(device), label_list[i].to(device, dtype=torch.int64)
            loss = criterion(torch.unsqueeze(outputs,0), labels)
            loss_sum_valid = loss_sum_valid + loss
            count_valid = count_valid+1
    print(f'[{epoch + 1}] loss of train_dataset: {loss_sum / count:.7f}')
    print(f'[{epoch + 1}] loss of valid_dataset: {loss_sum_valid / count_valid:.7f}')
    torch.save(net.state_dict(), 'deep_kws-'+str(epoch)+'.pth')

torch.save(net.state_dict(), 'deep_kws.pth')
print('Finished Training')
