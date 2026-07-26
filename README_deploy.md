## Installation Deploy using Conda

1. install [miniforge](https://github.com/conda-forge/miniforge/releases) (x86_64 or arm64 depending on your platform)

2. create an environment using the file in the folder [deploy/installation](./deploy/installation):


```bash
conda env create -f mamba_environment.yaml
conda activate fault_locomotion_isaaclab_env
```


## Run Sim-to-Sim 


```bash
## Sim-to-Sim
python3 deploy/play_mujoco.py


## Sim-to-Sim with ROS2
python3 deploy/run_controller_ros2.py (TERMINAL 1) 

python3 deploy/run_simulator_ros2.py (TERMINAL 2)


ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 3)

```

## Run Sim-to-Real

```bash
## Sim-to-Real with ROS2 (TERMINAL 1)
python3 deploy/run_controller_ros2.py (TERMINAL 1)

ros2 launch teleop_twist_joy teleop-launch.py joy_config:='xbox' (if want joystick) (TERMINAL 2)
```
