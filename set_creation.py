from tkinter import Label
import torch
import torchaudio as audio
import linecache

# Lable
pos_tag_path = "/home/disk1/user5/task1_data/positive/text"
neg_tag_path = "/home/disk1/user5/task1_data/negative/text"

# Train
pos_train_index_path = "/home/disk1/user5/task1_data/positive/train.scp"
neg_train_index_path = "/home/disk1/user5/task1_data/negative/train.scp"
train_target_path = "/home/disk1/user5/wyz/DataSet/TrainSet/"

# Test
pos_test_index_path = "/home/disk1/user5/task1_data/positive/test.scp"
neg_test_index_path = "/home/disk1/user5/task1_data/negative/test.scp"
test_target_path = "/home/disk1/user5/wyz/DataSet/TestSet/"

# Valid
pos_valid_index_path = "/home/disk1/user5/task1_data/positive/valid.scp"
neg_valid_index_path = "/home/disk1/user5/task1_data/negative/valid.scp"
valid_target_path = "/home/disk1/user5/wyz/DataSet/ValidSet/"

Labels = {"null": 0, "Hello": 1, "xiaogua": 2, "nihao": 3, "xiaoyi": 4, "jixu": 5, "tingzhi": 6, "bofang":7}
key_class = ("Hello", "xiaogua", "nihao", "xiaoyi", "jixu", "tingzhi", "bofang", "null")

Scaling = 1.6

def return_first_halfkey(whole_key):
    if whole_key.startswith(key_class[0]):
        return key_class[0]
    elif whole_key.startswith(key_class[2]):
        return key_class[2]
    elif whole_key.startswith(key_class[4]):
        return key_class[4]
    elif whole_key.startswith(key_class[5]):
        return key_class[5]
    else:
        return key_class[7]

def return_second_halfkey(whole_key):
    if whole_key.startswith(key_class[0]):
        return key_class[1]
    elif whole_key.startswith(key_class[2]):
        return key_class[3]
    elif whole_key.startswith(key_class[4]):
        return key_class[6]
    elif whole_key.startswith(key_class[5]):
        return key_class[6]
    else:
        return key_class[7]

def create_positive_set(index_path, tag_path, target_root_path):
    with open(index_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            items = line.split()
            name = items[0].split("-")
            video_path = items[1].replace("\n", "")

            time_label = linecache.getline(tag_path, 23852+i+1).strip().split()
            label = name[1]
            feature_name = label+"-"+time_label[1]+"-"+time_label[2]+"-"+time_label[3]+"-"+time_label[4]+"-"+name[0]+name[-1]

            waveform, sr = audio.load(video_path)
            fbank_tensor = audio.compliance.kaldi.fbank(waveform=waveform, frame_length=25.0, frame_shift=10.0, num_mel_bins=40)
            tensor_label = list(range(fbank_tensor.shape[0]))
            for j in range(fbank_tensor.shape[0]):
                if(j+1 > Scaling*int(time_label[1]) and j+1 < Scaling*int(time_label[2])):
                    tensor_label[j] = Labels.get(return_first_halfkey(label))
                elif(j+1 > Scaling*int(time_label[3]) and j+1 < Scaling*int(time_label[4])):
                    tensor_label[j] = Labels.get(return_second_halfkey(label))
                else:
                    tensor_label[j] = Labels.get("null")
            tensor_label = torch.tensor(tensor_label).unsqueeze(1)
            fbank_tensor = torch.cat((fbank_tensor, tensor_label),1)

            torch.save(fbank_tensor, target_root_path+feature_name+".pth")
            print("saved tensor to:" + target_root_path+feature_name+".pth")


def create_negative_set(index_path, tag_path, target_root_path):
    with open(index_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split()
            name = items[0]
            video_path = items[1].replace("\n", "")

            label = "null"
            feature_name = label+"-"+name

            waveform, sr = audio.load(video_path)
            fbank_tensor = audio.compliance.kaldi.fbank(waveform=waveform, frame_length=25.0, frame_shift=10.0, num_mel_bins=40)
            tensor_label = list(range(fbank_tensor.shape[0]))
            for j in range(fbank_tensor.shape[0]):
                tensor_label[j] = Labels.get("null")
            tensor_label = torch.tensor(tensor_label).unsqueeze(1)
            fbank_tensor = torch.cat((fbank_tensor, tensor_label),1)

            torch.save(fbank_tensor, target_root_path+feature_name+".pth")
            print("saved tensor to:" + target_root_path+feature_name+".pth")


#create_positive_set(index_path=pos_train_index_path, tag_path=pos_tag_path, target_root_path=train_target_path)
#create_negative_set(index_path=neg_train_index_path, tag_path=neg_tag_path, target_root_path=train_target_path)


#create_positive_set(index_path=pos_valid_index_path, tag_path=pos_tag_path, target_root_path=valid_target_path)
#create_negative_set(index_path=neg_valid_index_path, tag_path=neg_tag_path, target_root_path=valid_target_path)

#create_positive_set(index_path=pos_test_index_path, tag_path=pos_tag_path, target_root_path=test_target_path)
#create_negative_set(index_path=neg_test_index_path, tag_path=neg_tag_path, target_root_path=test_target_path)