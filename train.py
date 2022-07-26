
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

fbanks_batch_size = 1
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('using device:', device)


train_set = KWSDataSet(train_dataset_path)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=fbanks_batch_size, collate_fn=generate_batch_data,
                                            shuffle=True, num_workers=16, pin_memory=True)
valid_set = KWSDataSet(valid_dataset_path)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=fbanks_batch_size, collate_fn=generate_batch_data,
                                            shuffle=True, num_workers=16, pin_memory=True)


net = KWSNet2().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.0005)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.60)


def valid():
    count_valid = 0
    loss_sum_valid = 0.0
    correct = 0
    net.eval()
    with torch.no_grad():
        for index, data in enumerate(valid_loader):
                
            inputs = data[0].to(device)
            labels =data[1].to(device, dtype=torch.int64)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_sum_valid = loss_sum_valid + loss
            correct += (outputs.argmax(1)==labels).type(torch.float).sum().item()
            count_valid += labels.shape[0]
            
            del inputs, labels
            if index % 1000 == 999:
                print(f'average valid loss: {loss_sum_valid / (index+1):.7f}')
                print(f'acc: {correct / (count_valid):.7f}')

def train():
    for epoch in range(20): 
        net.train()
        print("learning rate of epoch-%d: %f" % (epoch, optimizer.param_groups[0]['lr']))
        loss_epoch_sum = 0.0

        for index, data in enumerate(train_loader):
            inputs = data[0].to(device)
            labels =data[1].to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = net(inputs)
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            del inputs, labels
            loss_epoch_sum = loss_epoch_sum + loss
            
            
            
        
        print(f'[{epoch + 1}] average train loss: {loss_epoch_sum / (index+1):.7f}')
        StepLR.step()
        print("learning rate of epoch-%d: %f" % (epoch, optimizer.param_groups[0]['lr']))
        
        
        if epoch % 4 == 3:
            torch.save(net.state_dict(), 'deep_kws-net3-'+str(epoch)+'.pth')

        valid()
    


if __name__ == '__main__':
    train()
    
    torch.save(net.state_dict(), 'deep_kws.pth')
    print('Finished Training')
