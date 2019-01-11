import tensorflow as tf
from tensorflow.contrib import learn

class Lstm(object):
    def build(self,input,input_size,rnn_size,num_class):
        x = input
        #将feature map进行处理

        dim = tf.shape(x)
        x = tf.reshape(x, [dim[0]*dim[1],dim[2] ,input_size])
        x.set_shape([None,None,input_size])
        x = self._gru_layer(x,rnn_size)
        x = tf.reshape(x,[-1, rnn_size*2])
        x = self._fc_layer(x,rnn_size*2,num_class)
        x= tf.reshape(x, [-1, dim[1], dim[2],num_class])

        return x

    def _fc_layer(self, bottom, in_size, out_size):
            initial_value = tf.random_normal([in_size, out_size], 0.0, 0.001)
            weights = tf.Variable(initial_value=initial_value)
            biases = tf.Variable(tf.zeros([out_size], dtype=tf.float32))
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
    def _gru_layer(self,input_sequence, rnn_size):

        cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
        cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
        rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    dtype=tf.float32)
        rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
        return rnn_output_stack

    def _rnn_layer(self,input_sequence, rnn_size):

        weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
        cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=weight_initializer)
        cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=weight_initializer)
        rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                        dtype=tf.float32)
        rnn_output_stack = tf.concat(rnn_output, 2, name='output_stack')
        return rnn_output_stack

if __name__ == "__main__":
    input_data = tf.placeholder(tf.float32, [5,8,8,128])
    rnn = Lstm()
    output_data = rnn.build(input_data,128,512)
    print(output_data)
