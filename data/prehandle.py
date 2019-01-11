from sys import argv
import os
import math
import cv2 as cv
import numpy as np
import cv2
import glob

class PreHandle(object):

    def my_sort(elem):
        p = elem.strip().split(' ')
        img = cv2.imread(elem)
        return img.shape[0] / img.shape[1]

    ##############################################
    #将图片按照宽高比进行排序
    #   params:
    #           image_path:图片的路径
    #           filename: 最后生成file的名称
    #############################################
    def generate_file(image_path,filename):
        with open(filename, 'w', encoding='utf-8') as f:
            files = glob.glob(image_path+'/*.jpg')
            files.sort(key=PreHandle.my_sort)
            for file in files:
                f.write(file.split('/')[-1]+'\n')

    ##################################################
    #  将所有的label中的text area 按照宽度16进行分割
    #   params:
    #           image_path:  图片的路径
    #           label_path:  label的路径
    #           new_image_path: 新生成的图片的路径
    #           new_label_path: 新生成的label的路径
    ##################################################
    def split_labels(image_path, label_path,new_image_path,new_label_path):
        files = os.listdir(image_path)
        files.sort()
        for file in files:
            _, basename = os.path.split(file)
            if basename.lower().split('.')[-1] not in ['jpg', 'png']:
                continue
            stem, ext = os.path.splitext(basename)
            gt_file = os.path.join(label_path, 'gt_' + stem + '.txt')
            img_path = os.path.join(image_path, file)

            img = cv.imread(img_path)
            img_size = img.shape

            im_size_min = np.min(img_size[0:2])
            im_size_max = np.max(img_size[0:2])
            im_scale = float(600) / float(im_size_min)
            if np.round(im_scale * im_size_max) > 1200:
                im_scale = float(1200) / float(im_size_max)
            re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
            re_size = re_im.shape

            if not os.path.exists(new_image_path):
                os.makedirs(new_image_path)
            cv.imwrite(os.path.join(new_image_path, stem) + '.jpg', re_im)

            with open(gt_file, 'r',encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                splitted_line = line.strip().lower().split(',')
                pt_x = np.zeros((4, 1))
                pt_y = np.zeros((4, 1))

                pt_x[0, 0] = int(float(splitted_line[0]) / img_size[1] * re_size[1])
                pt_y[0, 0] = int(float(splitted_line[1]) / img_size[0] * re_size[0])
                pt_x[1, 0] = int(float(splitted_line[2]) / img_size[1] * re_size[1])
                pt_y[1, 0] = int(float(splitted_line[3]) / img_size[0] * re_size[0])
                pt_x[2, 0] = int(float(splitted_line[4]) / img_size[1] * re_size[1])
                pt_y[2, 0] = int(float(splitted_line[5]) / img_size[0] * re_size[0])
                pt_x[3, 0] = int(float(splitted_line[6]) / img_size[1] * re_size[1])
                pt_y[3, 0] = int(float(splitted_line[7]) / img_size[0] * re_size[0])

                ind_x = np.argsort(pt_x, axis=0)
                pt_x = pt_x[ind_x]
                ind_y = np.argsort(pt_y,axis =0)
                pt_y = pt_y[ind_y]



                xmin = pt_x[0]
                ymin = pt_y[0]

                xmax = pt_x[3]
                ymax = pt_y[3]

                if xmin < 0:
                    xmin = 0

                if xmax > re_size[1] - 1:
                    xmax = re_size[1] - 1

                if ymin < 0:
                    ymin = 0

                if ymax > re_size[0] - 1:
                    ymax = re_size[0] - 1

                width = xmax - xmin
                height = ymax - ymin

                # reimplement
                step = 16.0
                x_left = []
                x_right = []
                x_left.append(xmin)
                x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
                if x_left_start == xmin:
                    x_left_start = xmin + 16

                for i in np.arange(x_left_start, xmax, 16):
                    x_left.append(i)

                x_left = np.array(x_left)
                x_right.append(x_left_start - 1)
                for i in range(1, len(x_left) - 1):
                    x_right.append(x_left[i] + 15)
                x_right.append(xmax)
                x_right = np.array(x_right)
                idx = np.where(x_left == x_right)
                x_left = np.delete(x_left, idx, axis=0)
                x_right = np.delete(x_right, idx, axis=0)
                if not os.path.exists(new_label_path):
                    os.makedirs(new_label_path)
                with open(os.path.join(new_label_path, stem) + '.txt', 'a') as f:
                    for i in range(len(x_left)):
                       # f.writelines("text\t")
                        f.writelines(str(int(x_left[i])))
                        f.writelines("\t")
                        f.writelines(str(int(ymin)))
                        f.writelines("\t")
                        f.writelines(str(int(x_right[i])))
                        f.writelines("\t")
                        f.writelines(str(int(ymax)))
                        f.writelines("\n")
def show():
    print('使用方法:')
    print('生成训练需要的数据: prehandle.py -g [src_iamge_folder] [src_label_folder] [new_image_folder] [new_label_folder]')
    print('根据宽高比进行排序:  prehandle.py -s [image_folder] [file]')
if __name__ == "__main__":
    ls = argv
    if(len(ls)!=6 and len(ls)!=4):
        show()
        exit()
    if(ls[1]!='-g' and ls[1]!='-s'):
        show()
        exit()
    if(ls[1]=='-g' and len(ls)==6):
        PreHandle.split_labels(ls[2],ls[3],ls[4],ls[5])
        exit()
    if(ls[1]=='-s' and len(ls)==4):
        PreHandle.generate_file(ls[2],ls[3])
        exit()
    show()