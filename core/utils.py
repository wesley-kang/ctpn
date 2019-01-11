import numpy as np
import cv2
import os
from ctypes import *
class Utils:
    def get_mean_wh(dir,ls_filename):
        sum_w = 0.
        sum_h = 0.
        for name in ls_filename:
            img = cv2.imread(os.path.join(dir,name))
            sum_h = sum_h + img.shape[0]
            sum_w = sum_w + img.shape[1]
        return sum_w//len(ls_filename),sum_h//len(ls_filename)
    ######################
    # 求取矩形框局部最大值
    ######################
    def c_nms(dets,thresh):
        c_dll = cdll.LoadLibrary('./core/c_utils.so')
        #将python中的类型转化为对应的ctype类型
        # create a pointer-array
        boxes = (POINTER(c_float) * dets.shape[0])()
        for i in range(dets.shape[0]):
            boxes[i] = (c_float * dets.shape[1])()

        for row in range(dets.shape[0]):
            for col in range(dets[row].shape[0]):
                boxes[row][col] = dets[row][col]
        thresh = c_float(thresh)
        rest_index  = (c_int*dets.shape[0])()
        rest_len = c_int(0)
        boxes_len = c_int(dets.shape[0])
        c_dll.nms(boxes,boxes_len,thresh,rest_index,pointer(rest_len))
        #将返回的结果转化为对应的python类型
        keep = []
        for index in range(rest_len.value):
            keep.append(rest_index[index])
        return keep


    ####################################
    #   将data中的数据重新映射会原数组
    ####################################
    def unmap(data, count, inds, fill=0):
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[inds] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[inds, :] = data
        return ret

    #########################################
    #   计算两组anchors之间的overlap(交并比)
    #########################################
    def c_cal_overlaps(anchors,gt_boxes):
        c_dll = cdll.LoadLibrary('./core/c_utils.so')

        len1 = anchors.shape[0]
        len2 = gt_boxes.shape[0]

        # create a pointer-array
        boxes1 = (POINTER(c_float) * len1)()
        for i in range(len1):
            boxes1[i] = (c_float * 4)()

        boxes2 = (POINTER(c_float) * len2)()
        for i in range(len2):
            boxes2[i] = (c_float * 4)()

        for row in range(len1):
            for col in range(4):
                boxes1[row][col] = anchors[row][col]
        for row in range(len2):
            for col in range(4):
                boxes2[row][col] = gt_boxes[row][col]

        ret = (POINTER(c_float) * len1)()
        for i in range(len1):
            ret[i] = (c_float * len2)()

        len1 = c_int(len1)
        len2 = c_int(len2)

        c_dll.cal_overlaps(boxes1,len1,boxes2,len2,ret)
        overlaps = np.zeros((len1.value, len2.value), dtype=np.float32)
        for row in range(len1.value):
            for col in range(len2.value):
                overlaps[row][col] = ret[row][col]
        return overlaps


    #################################
    # 每一个锚点，生成对应的10个锚框
    #################################

    def c_generate_all_anchors(batchsize,fea_w,fea_h,feat_stride):

        c_dll = CDLL('./core/c_utils.so')
        len = fea_w*fea_h*10*batchsize

        # create a pointer-array
        c_anchors = (POINTER(c_float)*len)()
        for i in range(len):
            c_anchors[i] = (c_float*4)()

        c_fea_w = c_int(fea_w)
        c_fea_h = c_int(fea_h)
        c_feat_stride = c_int(feat_stride)


        c_dll.generate_all_anchors(c_anchors,batchsize,c_fea_w,c_fea_h,c_feat_stride)

        anchors = np.zeros((len,4),dtype = np.float)
        for index in range(len):
            anchors[index][0] = c_anchors[index][0]
            anchors[index][1] = c_anchors[index][1]
            anchors[index][2] = c_anchors[index][2]
            anchors[index][3] = c_anchors[index][3]
        return anchors

    def generate_anchors():
        heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
        widths = [16]
        sizes = []
        for h in heights:
            for w in widths:
                sizes.append((h, w))
        return Utils.generate_basic_anchors(sizes)

    def generate_basic_anchors(sizes, base_size=16):
        base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
        anchors = np.zeros((len(sizes), 4), np.int32)
        index = 0
        for h, w in sizes:
            anchors[index] = Utils.scale_anchor(base_anchor, h, w)
            index += 1
        return anchors

    def scale_anchor(anchor, h, w):
        x_ctr = (anchor[0] + anchor[2]) * 0.5
        y_ctr = (anchor[1] + anchor[3]) * 0.5
        scaled_anchor = anchor.copy()
        scaled_anchor[0] = x_ctr - w / 2  # xmin
        scaled_anchor[2] = x_ctr + w / 2  # xmax
        scaled_anchor[1] = y_ctr - h / 2  # ymin
        scaled_anchor[3] = y_ctr + h / 2  # ymax
        return scaled_anchor



    ################################
    #   预测框到真实框之间的参数转化
    ################################

    def c_bbox_transform(ex_rois,gt_rois):
        c_dll = cdll.LoadLibrary('./core/c_utils.so')
        len = ex_rois.shape[0]
        c_ex_rois = ((4*c_float)*len)()
        c_gt_rois = ((4*c_float)*len)()
        c_targets = ((4*c_float)*len)()
        for index in range(len):
            c_ex_rois[index][0] = ex_rois[index][0]
            c_ex_rois[index][1] = ex_rois[index][1]
            c_ex_rois[index][2] = ex_rois[index][2]
            c_ex_rois[index][3] = ex_rois[index][3]

            c_gt_rois[index][0] = gt_rois[index][0]
            c_gt_rois[index][1] = gt_rois[index][1]
            c_gt_rois[index][2] = gt_rois[index][2]
            c_gt_rois[index][3] = gt_rois[index][3]
        c_len = c_int(len)
        c_dll.bbox_transform(c_ex_rois,c_gt_rois,c_targets,c_len)
        targets = np.zeros(shape=(len,4),dtype = np.float)
        for index in range(len):
            targets[index][0] = c_targets[index][0]
            targets[index][1] = c_targets[index][1]
            targets[index][2] = c_targets[index][2]
            targets[index][3] = c_targets[index][3]
        return targets



    ##########################################
    #   bbox_transform的逆操作
    ###########################################
    def c_bbox_transform_inv(boxes,deltas):
        c_dll = cdll.LoadLibrary('./core/c_utils.so')
        len = boxes.shape[0]



        # create a pointer-array
        c_boxes = (POINTER(c_float)*len)()
        for i in range(len):
            c_boxes[i] = (c_float*4)()

        c_deltas = (POINTER(c_float) * len)()
        for i in range(len):
            c_deltas[i] = (c_float * 4)()

        c_proposals = (POINTER(c_float) * len)()
        for i in range(len):
            c_proposals[i] = (c_float * 4)()


        for index in range(len):
            c_boxes[index][0] = boxes[index][0]
            c_boxes[index][1] = boxes[index][1]
            c_boxes[index][2] = boxes[index][2]
            c_boxes[index][3] = boxes[index][3]

            c_deltas[index][0] = deltas[index][0]
            c_deltas[index][1] = deltas[index][1]
            c_deltas[index][2] = deltas[index][2]
            c_deltas[index][3] = deltas[index][3]
        c_len = c_int(len)
        c_dll.bbox_transform_inv(c_boxes, c_deltas, c_proposals, c_len)
        proposals = np.zeros(shape=(len, 4), dtype=np.float)
        for index in range(len):
            proposals[index][0] = c_proposals[index][0]
            proposals[index][1] = c_proposals[index][1]
            proposals[index][2] = c_proposals[index][2]
            proposals[index][3] = c_proposals[index][3]
        return proposals


    def draw_boxes(img,image_name, boxes,scale = 1.0):
        for box in boxes:
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
        img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(image_name, img)


    def bbox_transform(ex_rois, gt_rois):

        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
        gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
        gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = np.log(gt_widths / ex_widths)
        targets_dh = np.log(ex_heights / ex_heights)

        targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
        return targets

    ##########################################
    #   bbox_transform的逆操作
    ###########################################
    def bbox_transform_inv(boxes, deltas):
        boxes = boxes.astype(deltas.dtype, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        pred_ctr_x = ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes
#a = np.array([[1,1,3,3],[2,2,4,4]])
#b = np.array([[3,3,4,4],[5,5,6,6]])
#c = Utils.c_cal_overlaps(a,b)
#print(c)
#
# for i in range(2):
#     for j in range(5):
#         print(b[i][j])
