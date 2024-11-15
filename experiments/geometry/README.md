# Introduction

Here we construct the experimental results for the geometry experiments in __Section 4__ of the paper. The working directory is `experiments/geometry/` and the experiments are executed by
```
python mapping.py [example] [lambda_reg]
```

+ `[example]` :
  + Simple cross example that works well with regularization: `cross`
  + Parameterization of the ant surface: `parameterization`
  + Scaling analysis in perturbed square: `corner`
+ `[lambda_reg]`: regularization parameter

There are additional settings in `mapping.py`. The QP is solved within `mapping_layer.py`, where the solver and other options can be modified.
To time the backward for dQP alone, and not the entire architecture, the lines inside `dQP.py` under `differentiate_QP` L1070 can be uncommented.