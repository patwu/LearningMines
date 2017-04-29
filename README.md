# LearningMines


## Installation

LearningMines requires the following dependencies:
* numpy
* tensorflow

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
mlp=MLP(n_input=2,n_output=1) #create a network, it has 2 input and 1 output
pred,loss=mlp.train(x,y)  #train the network with x and y, return the prediction of network and loss  
```
