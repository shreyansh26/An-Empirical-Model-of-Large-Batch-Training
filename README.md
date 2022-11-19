# An Empirical Model of Large-Batch Training

An approximate implementation of the OpenAI paper - [An Empirical Model of Large-Batch Training for MNIST](https://arxiv.org/abs/1812.06162). This is an approximate implementation because we do not have a multi-GPU setup and hence use sequential gradients of each step to calculate $B_{big}$ (refer Appendix A of the paper).

To calculate the simple noise scale $\mathcal{B}_{simple}$

```
python mnist_train.py --noise-scale --batch-size 128 --epochs 1 --lr 0.01
```

This gives an average noise scale value of $\approx$ 870 which is close to the value in the paper as well (900). Since MNIST's simple noise scale is an overestimate of the critical noise scale. $\mathcal{B}_{critical}$ (mentioned in the paper), we set the batch size to 512.

For model training,  

```
python mnist_train.py --batch-size 512 --epochs 25 --lr 0.01
```

Nice detailed reference [blogpost](https://towardsdatascience.com/implementing-a-batch-size-finder-in-fastai-how-to-get-a-4x-speedup-with-better-generalization-813d686f6bdf).