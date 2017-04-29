import tensorflow as tf
import numpy

float_type=tf.float32

def create_variable(name, shape, initializer, trainable=True):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=float_type, trainable=trainable)
    return var


class FCLayer(object):
    def __init__(self, layer_name, n_in, n_out, act=tf.nn.relu):
        self.n_in = n_in
        self.n_out = n_out
        self.act = act

        self.w = create_variable('%s-w' % layer_name, [n_in, n_out],
                                  tf.contrib.layers.xavier_initializer()) 
        self.b = create_variable('%s-b' % layer_name, [n_out],
                                      tf.constant_initializer(0.0, dtype=float_type))

    def forward(self, x, is_train=True):
        out=tf.matmul(x, self.w)
        out=out+self.b

        if self.act is None:
            return out
        else:
            return self.act(out)

