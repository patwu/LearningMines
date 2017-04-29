
from mlp import MLP

import numpy as np

if __name__=='__main__':
    
    #create network
    mlp=MLP(n_input=2,n_output=1)

    #training process
    interval=100
    correct=0
    for step in range(1000):
        #prepare training data
        x=np.random.randint(low=0,high=2,size=2)
        y=0
        for k in x:
           y=y^k
       
        #feed data to nerual network
        pred,loss=mlp.train(x,y)
        if (pred>0.5 and y==1) or (pred<0.5 and y==0):
            correct=correct+1

        if step %interval==0:
            #print 'training with x=%s y=%s'%(str(x),str(y))
            #print 'model pred=%s, loss=%s'%(str(pred),str(loss)) 
            print 'correct in last samples: (%d/%d)'%(correct,interval)
            correct=0
