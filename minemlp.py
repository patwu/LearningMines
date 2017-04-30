from board import Board
from mlp import MLP
import numpy as np
np.set_printoptions(linewidth=1000,precision=4)

mv=[[0,0],[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

def buildx(field,row,col,n_size):
    x=[]
    for i in range(9):
        nrow=row+mv[i][0]
        ncol=col+mv[i][1]
        if nrow in range(0,n_size) and ncol in range(0,n_size):
            t=[0]*10
            if field[nrow][ncol]==-1:
                t[9]=1
            else:
                t[field[nrow][ncol]]=1
        else:
            t=[0]*10
        x.append(t)
    x=np.asarray(x).flatten()
    return x

def train():
    mlp=MLP(n_input=9*10,n_output=2)
    n_size=10

    for e in range(100000):
        #eps-greedy
        board=Board(n_size,n_size,n_size) 
        while board.get_state()=='ready':
            field=board.get_array()
            eps=np.random.random()
            if eps<0.1:
                row, col = np.random.randint(0, n_size),np.random.randint(0, n_size)
                x=buildx(field,row,col,n_size)            
            else:
                best=-1
                bestact=None

                for row in range(n_size):
                    for col in range(n_size):
                        x=buildx(field,row,col,n_size)
                        _,prob=mlp.predict(x)
                        if prob[1]>best:
                            bestact=row,col
                            best=prob[1]
            
                row,col=bestact 
                x=buildx(field,row,col,n_size)
            board.play(row,col)
            
            if board.get_state()=='ready' or board.get_state()=='success':
                y=1
            else:
                y=0 
            _,loss=mlp.train(x,y)
            if e%100==0:
                print 'episode %d, loss: %.3f'%(e,loss)
        
        #try play
        if e%1000==0:
            board=Board(n_size,n_size,n_size)
            board.print_board()
            board.draw_board('tmp')
            while board.get_state()=='ready':
                field=board.get_array()
                best=-1
                bestact=None
                policy=np.zeros((n_size,n_size))
                for row in range(n_size):
                    for col in range(n_size):
                        x=buildx(field,row,col,n_size)
                        _,prob=mlp.predict(x)
                        policy[row][col]=prob[1]
                        if prob[1]>best:
                            bestact=row,col
                            best=prob[1]
            
                row,col=bestact 
                x=buildx(field,row,col,n_size)
                board.print_board()
                board.draw_board('tmp',policy)
                board.play(row,col)
            board.print_board()
            board.draw_board('tmp',policy)
 
            
if __name__=='__main__':
    train()    
