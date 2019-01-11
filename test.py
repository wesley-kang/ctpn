from core.resnet import ResNet
from core.lstm import Lstm
from core.loss import Loss
from core.datalayer import DataLayers
from core.proposallayer import ProposalLayer
from core.textdetector import TextDetector
from core.utils import Utils
from sys import argv
from config import Config
import tensorflow as tf
import numpy as np
import cv2
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 1

def fc_layer(x, num_in, num_out, name):
    dims = tf.shape(x)
    with tf.variable_scope(name):
        initial_value = tf.random_normal([num_in, num_out], 0.0, 0.001)
        weights = tf.Variable(initial_value=initial_value)
        biases = tf.Variable(tf.zeros([num_out], dtype=tf.float32))
        x = tf.reshape(x, [-1, num_in])
        x = tf.nn.bias_add(tf.matmul(x, weights), biases)
        x = tf.reshape(x, [dims[0], dims[1], dims[2], -1])
        return x

def build_graph():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')
    cnn_layers = ResNet()
    rnn_layers = Lstm()

    x = input_image
    x = cnn_layers.build(x, True)
    x = rnn_layers.build(x, 512,128, 512)

    ########################
    #   rpn cls score
    ########################

    y = fc_layer(x, 512, 10 * 2, "fc_rpn_cls")
    dims = tf.shape(y)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(y, [-1, 2])),
                          [dims[0], dims[1], -1, 2])

    #########################
    #   rpn bbox pred
    #########################
    box_pred = fc_layer(x, 512, 10 * 4, "fc_rpn_pred")

    return [input_image,cls_prob, box_pred]

def show():
    print('python test.py [image] [dst_image]')
if __name__ == "__main__":

    ls = argv
    if(len(ls)!=3):
        show()
        exit()
    tf_input = build_graph()
    input_images = tf_input[0]
    cls_prob = tf_input[1]
    box_pred = tf_input[2]
    image_info = {}
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        model_file = tf.train.latest_checkpoint('model/')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, model_file)

        img = cv2.imread(ls[1])
        im_scale = 1000/img.shape[0]
        resizeh = 1000
        resizew = int(img.shape[1]*im_scale)
        img = cv2.resize(img,(resizew,resizeh))

        image_info['image_data'] = img
        width = img.shape[1]
        height = img.shape[0]
        image_info['width'] = width
        image_info['height'] = height
        # 根据高和宽计算feature_map的大小
        if ( height% 16 == 0):
            feature_map_h = height // 16
        else:
            feature_map_h = height // 16 + 1
        if (width % 16 == 0):
            feature_map_w = width // 16
        else:
            feature_map_w = width // 16 + 1
        image_info['featuremap_h'] = feature_map_h
        image_info['featuremap_w'] = feature_map_w

        output_ls = sess.run([cls_prob, box_pred],feed_dict={input_images:img.reshape(-1,height,width,3)})
        cls = np.reshape(output_ls[0],(-1,2))
        box = np.reshape(output_ls[1],(-1,4))

        proposals = ProposalLayer.c_generate_proposals(cls, box,1,image_info, 16)
        proposals = proposals[proposals[:, 1] >= 0.7]
        boxes = TextDetector.c_detect(proposals[:,2:], proposals[:,1], [height,width])
        Utils.draw_boxes(img,ls[2], boxes)




