import numpy as np
import tensorflow as tf
class ResNet(object):
    def build(self, images, mode):

        self.mode = mode
        self.images = images
        return self._build_model()

    def _build_model(self):
        with tf.variable_scope('init'):
            x = self.images
            x = self._conv('init_conv', x, 3, 3, 32, [1,1,1,1])

        res_func = self._residual
        filters = [32,64, 128, 256,512]

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1],[1,2,2,1])
        for i in range(1, 3):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], [1,1,1,1])

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2],[1,2,2,1])
        for i in range(1, 4):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], [1,1,1,1])

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], [1,2,2,1])
        for i in range(1, 6):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], [1,1,1,1])

        with tf.variable_scope('unit_4_0'):
            x = res_func(x, filters[3], filters[4], [1,2,2,1])
        for i in range(1, 3):
            with tf.variable_scope('unit_4_%d' % i):
                x = res_func(x, filters[4], filters[4], [1,1,1,1])
        return x

    def _residual(self, x, in_filter, out_filter, stride):

        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = self._batch_norm( x)
            x = self._relu(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm(x)
            x = self._relu(x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'SAME')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride):

        with tf.variable_scope('residual_bn_relu'):
            orig_x = x
            x = self._batch_norm(x)
            x = self._relu(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm(x)
            x = self._relu(x)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self._batch_norm(x)
            x = self._relu(x)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x
        return x

    def _batch_norm(self, x):
        output = tf.contrib.layers.batch_norm(x,decay=0.9,
                                                epsilon=0.001,
                                                is_training=self.mode,
                                                fused=True,
                                                updates_collections=None)
        return output

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x):
        return tf.nn.relu(x)

    def _fully_connected(self, x, in_dim, out_dim):
        x = tf.reshape(x, [-1, in_dim])
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _max_pool(self, bottom):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return top
if __name__ == "__main__":
    input_data = tf.placeholder(tf.float32, [None, 128, 128, 3])
    cnn = ResNet()
    out = cnn.build(input_data,True)
    print(out)