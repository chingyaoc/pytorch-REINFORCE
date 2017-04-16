# PyTorch REINFORCE

<img src="assets/algo.png" width="800"> 

PyTorch implementation of REINFORCE.     
This repo supports both **continuous** and **discrete** environments in OpenAI gym. 


## Requirement
- python 2.7
- PyTorch
- OpenAI gym
- Mujoco (optional)


## Run
Use the default hyperparameters. *(Program will detect whether the environment is continuous or discrete)*

```
python main.py --env_name [name of environment]
```

## Experiment results
### continuous: InvertedPendulum-v1

<img src="assets/InvertedPendulum-v1.png" width="800">

### discrete: CartPole-v0

<img src="assets/CartPole-v0.png" width="800">

## Reference
- [pytorch example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)
