from core.utils import Utils
from config import Config
from ctypes import *
import numpy as np
import cv2
import os

class DataLayers(object):

    #初始化DataLayers层
    def __init__(self,path_images,path_labels,train_txt):
        with open(train_txt, 'r', encoding='utf-8') as f:
            self.ls_files = f.readlines()
            self.ls_files = [ file.strip('\n') for file in self.ls_files]
        self.index_of_files = 0
        self.num_files = len(self.ls_files)
        self.path_images = path_images
        self.path_labels = path_labels

    #每次从文件列表中获取一张图片,以及label信息
    def get_next(self,batchsize=8):
        image_info={}

        batch_images = self.ls_files[self.index_of_files:self.index_of_files+batchsize]
        self.index_of_files = (self.index_of_files + batchsize) %(self.num_files-batchsize)
        mean_w,mean_h = Utils.get_mean_wh(self.path_images,batch_images)
        resize_h = int(mean_h)
        resize_w = int(mean_w)
        #根据高和宽计算feature_map的大小
        if (resize_h % 16 == 0):
            feature_map_h = resize_h // 16
        else:
            feature_map_h = resize_h // 16 + 1
        if (resize_w % 16 == 0):
            feature_map_w = resize_w // 16
        else:
            feature_map_w = resize_w // 16 + 1


        img_rawdata = np.zeros(shape=(batchsize,resize_h,resize_w,3),dtype=np.float32)
        anchor_labels = np.zeros(shape=(batchsize,feature_map_h,feature_map_w*10,1),dtype=np.int32)
        anchor_target = np.zeros(shape=(batchsize,feature_map_h,feature_map_w*10,4),dtype=np.float32)
        inside_weight = np.zeros(shape=(batchsize,feature_map_h,feature_map_w*10,4),dtype=np.float32)
        outside_weight = np.zeros(shape=(batchsize,feature_map_h,feature_map_w*10,4),dtype=np.float32)


        for i,img_name in enumerate(batch_images):
            filename_label = img_name.split('.')[0]+'.txt'
            img = cv2.imread(os.path.join(self.path_images,img_name))
            im_scale = [resize_h/img.shape[0],resize_w/img.shape[1]]
            img = cv2.resize(img, (resize_w, resize_h))

            img_rawdata[i] = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
            with open(os.path.join(self.path_labels, filename_label), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            gt_boxes = np.zeros((len(lines), 4), dtype=np.float32)
            for x, line in enumerate(lines):
                l = line.split('\t')
                gt_boxes[x][0] = float(l[0])*im_scale[1]
                gt_boxes[x][1] = float(l[1])*im_scale[0]
                gt_boxes[x][2] = float(l[2])*im_scale[1]
                gt_boxes[x][3] = float(l[3])*im_scale[0]
            if(Config.USE_C):
                ret = self.c_gen_all_info(gt_boxes, feature_map_w, feature_map_h,resize_w,resize_h)
            else:
                ret = self.gen_all_info(gt_boxes,feature_map_w,feature_map_h)
            anchor_labels[i] = ret[0].reshape(1,feature_map_h,feature_map_w*10,1)
            anchor_target[i] = ret[1].reshape(1,feature_map_h,feature_map_w*10,4)
            inside_weight[i] = ret[2].reshape(1,feature_map_h,feature_map_w*10,4)
            outside_weight[i] = ret[3].reshape(1,feature_map_h,feature_map_w*10,4)
        image_info['anchor_labels'] = anchor_labels
        image_info['anchor_targets'] = anchor_target
        image_info['inside_weight'] = inside_weight
        image_info['outside_weight'] = outside_weight
        image_info['image_data'] = img_rawdata
        image_info['height'] = resize_h
        image_info['width'] = resize_w
        image_info['featuremap_h'] =  feature_map_h
        image_info['featuremap_w'] =  feature_map_w
        image_info['image_name']= batch_images

        return image_info
    def c_gen_all_info(self,gt_boxes,fea_w,fea_h,width,height):

        c_dll = cdll.LoadLibrary('./core/c_datalayer.so')

        # create a pointer-array
        c_gt_boxes = (POINTER(c_float) * gt_boxes.shape[0])()
        for i in range(gt_boxes.shape[0]):
            c_gt_boxes[i] = (c_float * gt_boxes.shape[1])()
            c_gt_boxes[i][0] = gt_boxes[i][0]
            c_gt_boxes[i][1] = gt_boxes[i][1]
            c_gt_boxes[i][2] = gt_boxes[i][2]
            c_gt_boxes[i][3] = gt_boxes[i][3]
        anchors_len  = 10*fea_w*fea_h

        # create a pointer-array
        c_ret = (POINTER(c_float) * anchors_len)()
        for i in range(anchors_len):
            c_ret[i] = (c_float * 13)()

        c_gt_len = c_int(gt_boxes.shape[0])
        c_fea_w  = c_int(fea_w)
        c_fea_h  = c_int(fea_h)
        c_width =  c_int(width)
        c_height = c_int(height)
        c_stride = c_int(16)


        c_dll.generate_labels_bboxes(c_gt_boxes,c_gt_len,c_fea_h,c_fea_w,c_height,c_width,c_stride,c_ret)

        labels = np.zeros((anchors_len,1),dtype = np.int32)
        bbox_targets =  np.zeros((anchors_len,4),dtype = np.float32)
        bbox_inside_weights = np.zeros((anchors_len,4),dtype = np.float32)
        bbox_outside_weights = np.zeros((anchors_len,4),dtype = np.float32)

        for i in range(anchors_len):

            labels[i][0] = c_ret[i][0]

            bbox_targets[i][0] = c_ret[i][1]
            bbox_targets[i][1] = c_ret[i][2]
            bbox_targets[i][2] = c_ret[i][3]
            bbox_targets[i][3] = c_ret[i][4]

            bbox_inside_weights[i][0] = c_ret[i][5]
            bbox_inside_weights[i][1] = c_ret[i][6]
            bbox_inside_weights[i][2] = c_ret[i][7]
            bbox_inside_weights[i][3] = c_ret[i][8]

            bbox_outside_weights[i][0] = c_ret[i][9]
            bbox_outside_weights[i][1] = c_ret[i][10]
            bbox_outside_weights[i][2] = c_ret[i][11]
            bbox_outside_weights[i][3] = c_ret[i][12]

        return  labels,bbox_targets,bbox_inside_weights,bbox_outside_weights


    #根据label信息，生成对应的anchor_label bbox_target
    def gen_all_info(self,gt_boxes,feature_map_w,feature_map_h):
        stride = 16
        ret_list = self.generate_labels_bboxs(gt_boxes,feature_map_h,feature_map_w,stride)
        return ret_list

    def generate_labels_bboxs(self,gt_boxes,feature_map_h,feature_map_w,stride):
        part_anchors,part_index,num_anchors = self._generate_all_anchors(feature_map_h,feature_map_w,stride)
        if(Config.USE_C):
            part_overlaps = Utils.c_cal_overlaps(part_anchors,gt_boxes)
        else:
            part_overlaps = Utils.cal_overlaps(part_anchors, gt_boxes)
        part_labels = self._cal_all_labels(part_overlaps)
        part_bbox_target = self._cal_all_box_targets(part_overlaps,part_anchors,gt_boxes)
        part_bbox_inside_weights = np.zeros((len(part_index), 4), dtype=np.float32)
        part_bbox_inside_weights[part_labels == 1, :] = np.ones((1,4))
        part_bbox_outside_weights = np.zeros((len(part_index), 4), dtype=np.float32)
        part_bbox_outside_weights[part_labels == 1, :] = np.ones((1, 4))
        # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
        labels = Utils.unmap(part_labels, num_anchors, part_index, fill=-1)
        bbox_targets = Utils.unmap(part_bbox_target, num_anchors, part_index, fill=0)  # 这些anchor的真值是0，也p即没有值
        bbox_inside_weights = Utils.unmap(part_bbox_inside_weights, num_anchors, part_index, fill=0)  # 内部权重以0填充
        bbox_outside_weights = Utils.unmap(part_bbox_outside_weights, num_anchors, part_index, fill=0)  # 外部权重以0填充

        labels = labels.reshape(-1,1)
        bbox_targets = bbox_targets.reshape(-1,4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1,4)
        bbox_outside_weights = bbox_outside_weights.reshape(-1,4)
        return labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

    def _cal_all_box_targets(self,overlaps,anchors,gt_boxes):
        argmax_overlaps = overlaps.argmax(axis=1)
        bbox_targets = Utils.bbox_transform(anchors, gt_boxes[argmax_overlaps, :])
        return bbox_targets

    def _cal_all_labels(self,overlaps):

        labels = np.empty((overlaps.shape[0],), dtype=np.int8)
        labels.fill(-1)

        #求每个anchors的最大overlap
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]

        #求每个gt_boxes最大的overlap
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]

        # positive 1 negitive 0  dontcare -1
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps>=0.7] = 1
        labels[max_overlaps<=0.3] = 0

        #对正负样本进行均衡,主要按照正例的个数，如果负例过多，减少负例
        num_pos_label = np.where(labels == 1)[0].shape[0]
        num_neg_label = np.where(labels == 0)[0].shape[0]
        neg_label = np.where(labels==0)[0]

        #随机去掉一些负样本
        if(num_pos_label<num_neg_label):
            disable_inds = np.random.choice(neg_label, size=(num_neg_label - num_pos_label), replace=False)
            labels[disable_inds] = -1
        return labels

    def _generate_all_anchors(self,feature_h,feature_w,stride):
        anchors = Utils.generate_anchors()
        shift_x = np.arange(0, feature_w) * stride
        shift_y = np.arange(0, feature_h) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        A = 10
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= 0) &
            (all_anchors[:, 1] >= 0) &
            (all_anchors[:, 2] < feature_w*stride) &  # width
            (all_anchors[:, 3] < feature_h*stride)  # height
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor
        return anchors,inds_inside,total_anchors
    def get_all_anchors(self,feature_h,feature_w,stride):
        anchors = Utils.generate_anchors()
        shift_x = np.arange(0, feature_w) * stride
        shift_y = np.arange(0, feature_h) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        A = 10
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        return all_anchors


if __name__ == "__main__":
    data_layer = DataLayers('../Data/new_train_image','../Data/new_train_label','../Data/train.txt')
    while(True):
        batch_data = data_layer.get_next(2)
        images_name = batch_data['image_name']
        batchsize = len(images_name)

        labels = batch_data['anchor_labels'].reshape(batchsize,-1,1)
        targets = batch_data['anchor_targets'].reshape(batchsize,-1,4)
        featuremap_h = batch_data['featuremap_h']
        featuremap_w = batch_data['featuremap_w']
        width = batch_data['width']
        height = batch_data['height']
        images_name = batch_data['image_name']
        images = batch_data['image_data']
        for index, name in enumerate(images_name):
            anchors = data_layer.get_all_anchors(feature_h=featuremap_h,feature_w=featuremap_w,stride=16)
            label = labels[index]
            target = targets[index]
            inds = np.where(label==1)[0]
            target = target[inds,:]
            anchors = anchors[inds,:]
            pros = Utils.bbox_transform_inv(anchors,target)
            img = images[index]
            img = img.reshape(height,width,3)
            Utils.draw_boxes(img,name,pros)


