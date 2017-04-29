
import tensorflow as tf
import numpy as np

from utils import FCLayer

class MLP(object):

    def __init__(self,n_input=2, n_hidden=5, n_output=1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output

        self.build()

    def loss(self,preds,labels):
        diff = tf.squared_difference(preds, labels)
        loss=tf.reduce_mean(diff)
        return loss

    def forward(self,x):
        h=FCLayer('layer1',n_in=self.n_input,n_out=self.n_hidden,act=tf.nn.tanh).forward(x)
        pred=FCLayer('layer2',n_in=self.n_hidden,n_out=self.n_output,act=tf.nn.sigmoid).forward(h)
        return pred

    def build(self):
        global_step = self.global_step= tf.Variable(0, name='global_step', trainable=False)

        x=self.x=tf.placeholder(tf.float32,[None,self.n_input])
        y=self.y=tf.placeholder(tf.float32,[None,self.n_output])

        self.pred_step=pred=self.forward(x)
        self.loss_step=loss=self.loss(pred,y)

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.train_step=opt.minimize(loss, global_step=global_step)
       
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self,x,y):
        x=np.reshape(np.asarray(x,dtype=np.float32),(1,self.n_input))
        y=np.reshape(np.asarray(y,dtype=np.float32),(1,self.n_output))

        feed = {self.x:x,self.y:y}
        _,loss,pred=self.sess.run([self.train_step,self.loss_step,self.pred_step], feed_dict=feed)
        return pred,loss

    def predict(self,x):
        x=np.reshape(np.asarray(x,dtype=np.float32),shape=(1,self.n_input))

        feed = {self.x:x}
        pred,loss=self.sess.run([self.pred_step,self.loss_step], feed_dict=feed)
        return pred[0],loss
       
