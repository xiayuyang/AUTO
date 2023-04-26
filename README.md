# AUTO : A Hierarchical Decision-making Framework with Multi-modality Perception for Autonomous Driving

![image](/figures/framework.png)

This repo is the implementation of the following paper:

**AUTO : A Hierarchical Decision-making Framework with Multi-modality Perception for Autonomous Driving**

<br> 

<br> 

## Code Structure
- algs<br>
Implementaion for algorithims.
- gym_carla<br>
Gym-like carla environment for vehicle agent controlled by reinforcement learning.
- main
    - tester<br>
    Code for testing model performance.
    - trainer<br>
    Code for training reinforcement learning model.
## Getting started
1. Install and setup [the CARLA simulator (0.9.14)](https://carla.readthedocs.io/en/latest/start_quickstart/#a-debian-carla-installation), set the executable CARLA_PATH in gym_carla/setting.py

2. Setup conda environment with cuda 11.7
```shell
$ conda create -n env_name python=3.7
$ conda activate env_name
```
3. Clone the repo and Install the dependent package
```shell
$ git clone https://github.com/greenday12138/AUTO.git
$ pip install -r requirements.txt
```
4. Train the RL agent in the multi-lane scenario
```shell
$ python ./main/trainer/pdqn_multi_process.py
```
4. Test the RL agent in the multi-lane scenario
```shell
$ python ./main/tester/multi_lane_test.py
```

## Training performance
![image](/figures/curve1.png)
![image](/figures/curve2.png)

(a-b) Results in the multi-lane scenario;
(c-d) Results in the crossing scenario.

## Reference


## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements
Our code is based on several repositories:
- [gym-carla](https://github.com/cjy1992/gym-carla.git)
- [CARLA_Leaderboard](https://github.com/RobeSafe-UAH/CARLA_Leaderboard.git)
- [MP-DQN](https://github.com/cycraig/MP-DQN.git)