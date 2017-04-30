# LearningMines


## Installation

LearningMines requires the following dependencies:
* numpy
* tensorflow
* Pillow
Clone the repo and install with pip.

```bash
git clone https://github.com/patwu/LearningMines.git
```

## XOR with mlp

Train a network to calcuate XOR operation

```bash
python xor.py
```
The network can feed data of different task to calcuate AND, OR. Modify the n_output to implement multi-bit, counting the number of 1

Reference api:

```python
np.random.randint(low=0,high=2,size=2) #generate a 1 dimension array with size of 2, each element in array is a random integer between 0 and 1
mlp=MLP(n_input=2,n_output=2) #create a network, it has 2 input and 2 types of output
pred,loss=mlp.train(x,y)  #train the network with x and y, return the prediction of network and loss  
```

## Mines AI with mlp

Train a mlp to predict the success rate of play by board state around move

```bash
python minemlp.py
```
For each coord in board, collecting the 3-3 states around and encoding the the state to 1-hot vector. Predicting the probability of success move by mlp as before.

Traing the mlp by eps-greedy strategy

Reference api:

```python
np.flatten() #flatten a multi dimension array to a 1-dimension array
np.asarray() #convert a python list to a np array
mlp.pred(x)  #predict using mlp
board.print_board() #show board at concole
board_draw_board(path,policy) #show board using png
board.play(row,col) #play a move at row,col
```


