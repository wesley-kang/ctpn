from core.proposallayer import ProposalLayer
from core.datalayer import DataLayers
from core.resnet import ResNet
from core.lstm import Lstm
from core.loss import Loss
from core.textdetector import TextDetector
from config import Config
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']=Config.GPU_DEVICE_NUM
batch_size = Config.BATCHSIZE
iterations =  Config.TOTAL_SAMPLES//Config.BATCHSIZE
total_epoch = Config.TOTAL_EPOCHS

def fc_layer(x,num_in,num_out,name):
    dims = tf.shape(x)
    with tf.variable_scope(name):
        initial_value = tf.random_normal([num_in, num_out], 0.0, 0.001)
        weights = tf.Variable(initial_value=initial_value)
        biases = tf.Variable(tf.zeros([num_out], dtype=tf.float32))
        x = tf.reshape(x, [-1, num_in])
        x = tf.nn.bias_add(tf.matmul(x, weights), biases)
        x = tf.reshape(x,[dims[0],dims[1],dims[2],-1])
        return  x

def build_graph():
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')
    input_label = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
    input_bbox_targets = tf.placeholder(tf.float32, shape=[None, 4], name='gt_boxes')
    input_bbox_inside_weights = tf.placeholder(tf.float32, shape=[None, 4], name="bbox_inside_weights")
    input_bbox_outside_weights = tf.placeholder(tf.float32, shape=[None, 4], name='bbox_outside_weights')
    learing_rate = tf.placeholder(tf.float32)

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

    loss_layer = Loss()
    output_loss ,cls_loss,box_loss = loss_layer.build(y,box_pred,input_label,input_bbox_targets,input_bbox_inside_weights,input_bbox_outside_weights)
    train_step = tf.train.AdamOptimizer(learing_rate).minimize(output_loss)

    return [train_step,output_loss,learing_rate,input_image,
            input_label,input_bbox_targets,input_bbox_inside_weights,input_bbox_outside_weights,cls_prob,box_pred,cls_loss,box_loss]

if __name__ == "__main__":

    tf_input = build_graph()
    train_data_layer = DataLayers(Config.TRAIN_IMAGE_PATH,Config.TRAIN_LABEL_PATH,Config.TRAIN_IMAGE_FILE)
    test_data_layer = DataLayers(Config.TEST_IMAGE_PATH,Config.TEST_LABLE_PATH, Config.TEST_IMAGE_FILE)
    train_step = tf_input[0]
    output_loss = tf_input[1]
    learing_rate = tf_input[2]
    input_images = tf_input[3]
    input_labels = tf_input[4]
    input_bbox_targets = tf_input[5]
    input_bbox_inside_weights = tf_input[6]
    input_bbox_outside_weights = tf_input[7]
    cls_prob = tf_input[8]
    box_pred = tf_input[9]
    cls_loss=tf_input[10]
    box_loss = tf_input[11]
    lr = Config.LEARNING_RATE

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        model_file = tf.train.latest_checkpoint('model/')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if (Config.RESTORE):
            saver.restore(sess, model_file)
        for ep in range(1,total_epoch+1):
            pre_index = 0.
            train_fscore = 0.
            train_loss = 0.

            for it in range(1, iterations + 1):
              
                image_info = train_data_layer.get_next(batch_size)
                output_ls = sess.run([train_step,output_loss,cls_prob,box_pred,cls_loss,box_loss],
                                         feed_dict={input_images: image_info['image_data'],
                                                    input_labels: image_info['anchor_labels'].reshape(-1,1),
                                                    input_bbox_targets:image_info['anchor_targets'].reshape(-1,4),
                                                    input_bbox_inside_weights:image_info['inside_weight'].reshape(-1,4),
                                                    input_bbox_outside_weights:image_info['outside_weight'].reshape(-1,4),
                                                    learing_rate: lr})
                print("iter:{} epoch {}/{} total loss:{} cls loss{} box loss{}".format(it, ep, total_epoch, output_ls[1],output_ls[4],output_ls[5]))


                if(it%Config.ITERS_PER_DECAY== 0):
                    lr = lr *Config.DECAY
                    image_info = test_data_layer.get_next(batch_size)
                    output_ls = sess.run([output_loss, cls_prob, box_pred, cls_loss, box_loss],
                                         feed_dict={input_images: image_info['image_data'],
                                                    input_labels: image_info['anchor_labels'].reshape(-1, 1),
                                                    input_bbox_targets: image_info['anchor_targets'].reshape(-1, 4),
                                                    input_bbox_inside_weights: image_info['inside_weight'].reshape(-1,4),
                                                    input_bbox_outside_weights: image_info['outside_weight'].reshape(-1,4),
                                                    learing_rate: lr})
                    cls = np.reshape(output_ls[1], (-1, 2))
                    box = np.reshape(output_ls[2], (-1, 4))
                    proposals = ProposalLayer.c_generate_proposals(cls,box,batch_size,image_info,16)
                    ret = ProposalLayer.cal_evaluation_index(Config.TEST_IMAGE_PATH,Config.TEST_LABLE_PATH,proposals,image_info)
                    print('precision:{} recall:{} fscore:{}'.format(ret[0],ret[1],ret[2]))
                    saver.save(sess, 'model/ctpn.ckpt', global_step=(ep - 1) * iterations + it)




