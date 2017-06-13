
from mlp import MLP

import numpy as np

if __name__=='__main__':
    
    #create network
    mlp=MLP(n_input=2,n_output=2)

    #training process
    interval=100
    correct=0
    for step in range(10000):
        #prepare training data
        x=np.random.randint(low=0,high=2,size=2)
        y=0
        for k in x:
           y=y^k
       
        #feed data to nerual network
        pred,loss=mlp.train(x,y)
        pred,prob=mlp.predict(x)
        if(pred==y):
            correct=correct+1

        if step %interval==0:
            #print 'training with x=%s y=%s'%(str(x),str(y))
            #print 'model pred=%s, loss=%s'%(str(pred),str(loss)) 
            print 'correct in last samples: (%d/%d)'%(correct,interval)
            correct=0
