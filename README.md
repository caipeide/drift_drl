# High-speed Autonomous Drifting with Deep Reinforcement Learning

<div align=center> 
<img src="./images/3.gif" width=270 alt="High-speed drifting cornering by the proposed deep RL controller"/>
<img src="./images/2.gif" width=270 alt="High-speed drifting cornering by the proposed deep RL controller"/>
<img src="./images/1.gif" width=270 alt="High-speed drifting cornering by the proposed deep RL controller"/>
</div>

## Reference trajectorires for seven maps
<div align=center> <img src="./images/maps.png" alt="Seven maps designed in this work"/>
</div>

Reference trajectories for the maps are located in `code/ref_trajectory`

**traj_0**: for map(a), for first-stage training.

**traj_1...traj_5**: for map(b-f), for second-stage training.

**traj_6**: for map(g), for evaluation

## Trained weights for different models
weights are located in `weights/`, where four kinds of models are included. Note that `sac-stg1` and `sac-stg2` are different stages of our SAC controller during training. `sac-stg2` is the final version and `sac-stg1` are only trained on map(a).

## Test code
run `sh code/test.sh` to test different models on map(g) with various setups. The driving data (timestamp, speed, location, heading, slip angle, control commands, etc.) will be recorded in `code/test/` after the testing process.

## Citation

Please consider to cite our paper if this work helps:
```
@ARTICLE{8961997,
author={P. {Cai} and X. {Mei} and L. {Tai} and Y. {Sun} and M. {Liu}},
journal={IEEE Robotics and Automation Letters},
title={High-Speed Autonomous Drifting With Deep Reinforcement Learning},
year={2020},
volume={5},
number={2},
pages={1247-1254},
keywords={Deep learning in robotics and automation;field robots;motion control;deep reinforcement learning;racing car},
doi={10.1109/LRA.2020.2967299},
ISSN={2377-3774},
month={April},}
```

[Project homepage](https://sites.google.com/view/autonomous-drifting-with-drl)
