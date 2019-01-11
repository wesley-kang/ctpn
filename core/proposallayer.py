from core.utils import Utils
from config import Config
import numpy as np
import os
import cv2
from ctypes import *
class ProposalLayer:
    def cal_evaluation_index(path_images,path_labels,proposals,image_info):
        precision = 0.
        recall = 0.
        fscore = 0.
        images_name = image_info['image_name']
        width = image_info['width']
        height = image_info['height']
        for index,name in enumerate(images_name):
            img = cv2.imread(os.path.join(path_images, name))
            im_scale = [height/img.shape[0], width/img.shape[1]]
            filename_label = name.split('.')[0] + '.txt'
            with open(os.path.join(path_labels, filename_label), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            gt_boxes = np.zeros((len(lines), 4), dtype=np.float32)
            for x, line in enumerate(lines):
                l = line.split('\t')
                gt_boxes[x][0] = float(l[0]) * im_scale[1]
                gt_boxes[x][1] = float(l[1]) * im_scale[0]
                gt_boxes[x][2] = float(l[2]) * im_scale[1]
                gt_boxes[x][3] = float(l[3]) * im_scale[0]
            inds = np.where(proposals[:,0]==index)[0]

            proposal = proposals[inds,:]

            if(Config.USE_C):
                overlaps = Utils.c_cal_overlaps(proposal[:,2:],gt_boxes)
            else:
                overlaps = Utils.cal_overlaps(proposals[:,2:],gt_boxes)
            # 求每个proposal的最大overlap
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
            max_overlaps = max_overlaps[max_overlaps>0.7]

            precision = precision + max_overlaps.shape[0]/overlaps.shape[0]
            recall = recall + max_overlaps.shape[0]/gt_boxes.shape[0]
        precision = precision/len(images_name)
        recall = recall/len(images_name)
        fscore = 2*precision*recall/(precision+recall+0.0001)
        return precision,recall,fscore

    def c_generate_proposals(rpn_cls_prob,rpn_bbox_pred,batchsize,image_info,feat_stride,post_nms_topn = 2000):
        c_dll = CDLL('./core/c_proposallayer.so')

        len = rpn_cls_prob.shape[0]
        # create a pointer-array
        c_cls = (POINTER(c_float) * len)()
        c_box_deltas = (POINTER(c_float) * len)()
        rlen = int(batchsize*post_nms_topn)
        c_ret = (POINTER(c_float) * int(rlen))()
        c_len = c_int(len)
        c_batchsize = c_int(batchsize)
        c_fea_w = c_int(image_info['featuremap_w'])
        c_fea_h = c_int(image_info['featuremap_h'])
        c_width = c_int(image_info['width'])
        c_height = c_int(image_info['height'])
        c_feat_stride = c_int(feat_stride)

        for i in range(len):
            c_cls[i] = (c_float * 2)()
            c_box_deltas[i] = (c_float*4)()

            c_cls[i][0] = rpn_cls_prob[i][0]
            c_cls[i][1] = rpn_cls_prob[i][1]

            c_box_deltas[i][0] = rpn_bbox_pred[i][0]
            c_box_deltas[i][1] = rpn_bbox_pred[i][1]
            c_box_deltas[i][2] = rpn_bbox_pred[i][2]
            c_box_deltas[i][3] = rpn_bbox_pred[i][3]

        for i in range(rlen):
            c_ret[i] =  (c_float*6)()
        rlen = c_int(rlen)
        c_dll.generate_proposals(c_cls,c_box_deltas,c_len,c_batchsize,c_fea_w,c_fea_h,c_width,c_height,c_feat_stride,c_ret,pointer(rlen))

        ret = np.zeros((rlen.value,6),dtype = np.float)
        for i in range(ret.shape[0]):
            ret[i][0] = c_ret[i][0]
            ret[i][1] = c_ret[i][1]
            ret[i][2] = c_ret[i][2]
            ret[i][3] = c_ret[i][3]
            ret[i][4] = c_ret[i][4]
            ret[i][5] = c_ret[i][5]
        return ret
