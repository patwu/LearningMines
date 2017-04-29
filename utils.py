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
                                  tf.random_normal_initializer(mean=0.0, stddev=2.0/numpy.sqrt(n_in), dtype=float_type))
        self.b = create_variable('%s-b' % layer_name, [n_out],
                                      tf.constant_initializer(0.1, dtype=float_type))

    def forward(self, x, is_train=True):
        out=tf.matmul(x, self.w)
        out=out+self.b

        if self.act is None:
            return out
        else:
            return self.act(out)

