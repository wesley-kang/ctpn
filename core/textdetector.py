import numpy as np
from core.utils import Utils

from ctypes import *

class TextDetector:

    def c_detect(text_proposals,scores,size):
        c_dll = cdll.LoadLibrary('./core/c_textproposalconnector.so')
        #返回值声明
        len = text_proposals.shape[0]
        box = (c_float * 9)
        boxes = (box * len)()

        #将python中的对象转化为c语言接受的类型
        proposals = ((c_float*4)*len)()
        scos = (c_float * len)()

        for i in range(len):
            proposals[i][0] = text_proposals[i,0]
            proposals[i][1] = text_proposals[i,1]
            proposals[i][2] = text_proposals[i,2]
            proposals[i][3] = text_proposals[i,3]
            scos[i] = scores[i]


        len = c_int(len)
        r_len = c_int()
        img_h = c_int(size[0])
        img_w = c_int(size[1])
        c_dll.get_text_lines(proposals, scos, len, img_h, img_w,boxes,pointer(r_len));
        #将返回结果转化为python的类型
        text_recs = np.zeros((r_len.value,9),dtype=np.float)
        for index in range(text_recs.shape[0]):
            text_recs[index,0] = boxes[index][0]
            text_recs[index,1] = boxes[index][1]
            text_recs[index,2] = boxes[index][2]
            text_recs[index,3] = boxes[index][3]
            text_recs[index,4] = boxes[index][4]
            text_recs[index,5] = boxes[index][5]
            text_recs[index,6] = boxes[index][6]
            text_recs[index,7] = boxes[index][7]
            text_recs[index,8] = boxes[index][8]
        return text_recs
