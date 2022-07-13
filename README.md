# NSDE
This is my PyTorch implementation for the paper:

> Wang, S. , &  Hong, L. J. . (2021). Option pricing by stochastic differential equations: A simulation optimization approach.

## Introduction

Here is the example of **50ETFcall option** dataset.

```
Best Iter=[10]@[172.9]	 test_MAE=[0.011914609879762134]
```

Hope it can help you.

## Environment Requirement

The code has been tested under Python 3.6.7. The required packages are as follows:

- pytorch == 1.10.2
- numpy == 1.19.5
- scipy == 1.1.0
- pandas == 1.0.5
- pyDOE == 0.3.8

## Example to Run the Codes

The instruction of commands has been clearly stated in the codes (see the parser function in Utility/parser.py).

```
python main.py --lr 1e-3 --save_flag 1 --epoch 400 --train_rate 0.8
```

