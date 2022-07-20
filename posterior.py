
from pip import main


class Posterior():
    def smooth_frame(outputs, output_length, w_smooth):
        
        for index, item in enumerate(outputs):
            if(index < w_smooth):
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
    
    '''
    def get_confidence_list(outputs, expect_labelindex_list, w_max):
        max_confidence_list = []
            
        for index, item in enumerate(outputs):
            if(index < w_max):
                
            else:
    '''
        
        
