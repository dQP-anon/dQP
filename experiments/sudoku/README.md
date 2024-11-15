# Introduction

Here we construct the experimental results for the Sudoku experiments adapted from OptNet in __Section 4__ of the paper. The working directory is `experiments/sudoku/` and the experiments are executed by
```
python train.py --boardSz 2 --nEpoch 20  dQPEq
```

+ `--boardSz` n corresponds to an nxn Sudoku puzzle. 2,3, and 4 are available.
+ `--nEpoch`: number of training epochs
+ `[model]`: dQPEq or optnetEq, which correspond to dQP and OptNet, respectively, in the equality constrained configuration of OptNet's original formulation. 

There are many additional settings in the original Sudoku experiment which can be located inside `train.py`. We choose some defaults, such as fixing the experiment to be on CPU. 
Note, however, that OptNet supports GPU batching which is well suited to many small problems. Results are located in the `work` directory.