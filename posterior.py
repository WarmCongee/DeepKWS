
import numpy as np


class Posterior():
    @staticmethod
    def smooth_frame(outputs, output_length, w_smooth):
        
        for index, item in enumerate(outputs):
            if(index < w_smooth-1):
                w_sum = [0.0]*output_length
                for i in range(index+1):
                    w_sum=[w_sum[n]+outputs[i][n] for n in range(min(len(w_sum),len(outputs[i])))]
                outputs[index] = [x/(index+1) for x in w_sum]
            else:
                w_sum = [0.0]*output_length
                for i in range(w_smooth):
                    w_sum=[w_sum[n]+outputs[index-i][n] for n in range(min(len(w_sum),len(outputs[index-i])))]
                outputs[index] = [x/(w_smooth) for x in w_sum]
        return outputs
    
    @staticmethod
    def get_max_list(outputs, output_length, w_max):
        max_list = []

        for index, item in enumerate(outputs):
            if(label-w_max+1 < 0):
                h_max = 0
            else:
                h_max = index-w_max+1
            max = [0.0]*output_length
            for label in range(len(outputs[0])):
                
                for i in range(h_max,index+1):
                    if max[label] < outputs[i][label]:
                        max[label] = outputs[i][label]
            max_list = max_list + max
        return max_list
    
    @staticmethod 
    def get_confidence(outputs, w_smooth, w_max):
        confidence = [0.0]*4
        outputs = Posterior.smooth_frame(outputs, outputs.shape[0], w_smooth)
        outputs = Posterior.get_max_list(outputs, outputs.shape[0], w_max)
        
        temp_conf = [0.0]*4
        for index, item in enumerate(outputs):
            if((item[1]*item[2]) ** 0.5 > confidence[0]):
                confidence[0] = (item[1]*item[2]) ** 0.5
            if((item[3]*item[4]) ** 0.5 > confidence[1]):
                confidence[1] = (item[3]*item[4]) ** 0.5
            if((item[5]*item[7]) ** 0.5 > confidence[2]):
                confidence[2] = (item[5]*item[7]) ** 0.5
            if((item[6]*item[7]) ** 0.5 > confidence[3]):
                confidence[3] = (item[6]*item[7]) ** 0.5
        return confidence
    
        
random_list = np.random.randint(10,size=[10,8])
print(random_list)

print("----------------------------------")

random_list = Posterior.smooth_frame(random_list, 8, 5)
print(random_list)