# tensorflow-vrnn
A variational recurrent neural network as described in:

[Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., & Bengio, Y. (2015). A recurrent latent variable model for sequential data. In Advances in neural information processing systems (pp. 2980-2988).](https://arxiv.org/abs/1506.02216)

## Requirements
python == 3.5
tensorflow == 1.2.1
numpy==1.13.1

## main.py
* train this model
```python
python main.py
```
## cell.py
* **VRNNCell** structure

## utils.py
* Basic functions implementation

## ops.py
* Basic operations based on tensorflow

## config.py
* Basic configuration of model
* Every configuration can be changed here.

![VRNN Structure](graph1.png?raw=true "VRNN Structure")
![Global Structure](graph2.png?raw=true "Global Structure")
