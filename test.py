
key_class = ("Hello", "xiaogua", "nihao", "xiaoyi", "jixu", "tingzhi", "bofang", "null")

# 关键字起止，向下取整，最小单位为fbank的40唯Tensor
first_key_start = 0
first_key_end = 0
second_key_start = 0
second_key_end = 0

first_key = ""
second_key = ""

def ini_key_se():
    first_key_start = 0
    first_key_end = 0
    second_key_start = 0
    second_key_end = 0

def set_key_se(first_key_start_, first_key_end_, second_key_start_, second_key_end_):
    first_key_start = first_key_start_
    first_key_end = first_key_end_
    second_key_start = second_key_start_
    second_key_end = second_key_end_

def return_first_halfkey(whole_key):
    if whole_key.startswith(key_class[0]):
        return key_class[0]
    elif whole_key.startswith(key_class[2]):
        return key_class[2]
    elif whole_key.startswith(key_class[4]):
        return key_class[4]
    else:
        return key_class[6]
def return_second_halfkey(whole_key):
    if whole_key.startswith(key_class[0]):
        return key_class[1]
    elif whole_key.startswith(key_class[2]):
        return key_class[3]
    elif whole_key.startswith(key_class[4]):
        return key_class[5]
    else:
        return key_class[7]

list = [1,5,6,6]

for i,data in enumerate(list):
    print(i)