
import tensorflow as tf
import numpy as np

from utils import FCLayer

class MLP(object):

    def __init__(self,n_input=2, n_hidden=16, n_output=2):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output

        self.build()

    def loss(self,logits,labels):
        ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        loss=tf.reduce_mean(ce)
        return loss

    def forward(self,x,dropout=True):
        h=FCLayer('layer1',n_in=self.n_input,n_out=self.n_hidden).forward(x)
#        if dropout:
#            h=tf.nn.dropout(h,0.5)
        h=FCLayer('layer2',n_in=self.n_hidden,n_out=self.n_hidden).forward(h)
#        if dropout:
#            h=tf.nn.dropout(h,0.5)
#        h=FCLayer('layer3',n_in=self.n_hidden,n_out=self.n_hidden).forward(h)
#        if dropout:
#            h=tf.nn.dropout(h,0.5)
#        h=FCLayer('layer4',n_in=self.n_hidden,n_out=self.n_hidden).forward(h)
#        if dropout:
#            h=tf.nn.dropout(h,0.5)
        logit=FCLayer('layer5',n_in=self.n_hidden,n_out=self.n_output, act=None).forward(h)
        pred=tf.nn.softmax(logit)

        tf.get_variable_scope().reuse_variables()
        return logit,pred

    def build(self):
        global_step = self.global_step=tf.Variable(0, name='global_step', trainable=False)

        x=self.x=tf.placeholder(tf.float32,[None,self.n_input])
        y=self.y=tf.placeholder(tf.int64,[None])

        logit,_=self.forward(x)
        self.loss_step=loss=self.loss(logit,y)

        _,pred=self.forward(x,dropout=False)
        self.prob_step=pred
        self.pred_step=tf.argmax(pred,axis=1)

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        self.train_step=opt.minimize(loss, global_step=global_step)
       
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self,x,y):
        x=np.reshape(np.asarray(x,dtype=np.float32),(1,self.n_input))
        y=np.reshape(y,(1,))

        feed = {self.x:x,self.y:y}
        _,loss,pred=self.sess.run([self.train_step,self.loss_step,self.pred_step], feed_dict=feed)
        return pred,loss

    def predict(self,x):
        x=np.reshape(np.asarray(x,dtype=np.float32),(1,self.n_input))

        feed = {self.x:x}
        pred,prob=self.sess.run([self.pred_step,self.prob_step], feed_dict=feed)
        return pred[0],prob[0]
       
