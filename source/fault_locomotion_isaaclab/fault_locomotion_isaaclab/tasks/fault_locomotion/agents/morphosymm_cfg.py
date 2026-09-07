from isaaclab.utils.configclass import configclass

from pathlib import Path
from dataclasses import MISSING

@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for using morphosymm-rl."""

    class_name: str = "MorphologycalSymmetries"
    """The class name."""

    obs_space_names_actor =  None
    """The observation space names for the actor network."""

    obs_space_names_critic = None
    """The observation space names for the critic network."""

    action_space_names = None
    """The action space names."""

    joints_order = None
    """The order of the joints in the robot."""

    robot_name = None
    """The name of the robot to use inside Morphosymm."""

    schedule_fixed_to_adaptive_switch = None
    """The number of iterations to switch from fixed to adaptive schedule for the symmetry loss.
    If None, then no switch will happen. If the scheduler is set to adaptive, not change will be made."""

    obs_space_names_single_state: list[str] | None = MISSING
    """The observation space names for the single state representation. This is used for DAE latent augmentation. If None, then no single state representation will be used."""


# Actor OBS
history_length = 5
single_state_names = [
        "base_lin_vel",
        "base_ang_vel",
        "gravity",
        "des_base_lin_vel_xy",
        "des_base_ang_vel_yaw",
        "joints_pos",
        "joints_vel",
        "joints_pos",
        #"clock_data",
        "clock_data", "clock_data", "clock_data",  # hip/thigh/calf statuses
    ]
obs_space_names_actor = single_state_names * int(history_length)
obs_space_names_actor += ["invariant_scalar"]


# Critic OBS
obs_space_names_critic = single_state_names * int(history_length)
obs_space_names_critic += [
        "clock_data", "clock_data", "clock_data",  # P gains
        "clock_data", "clock_data", "clock_data",  # D gains
]
obs_space_names_critic += [
        "base_lin_vel",  # clean lin vel b
        "invariant_scalar", "invariant_scalar",  # height error, terrain pitch
        "clock_data",  # contacts foot
        "clock_data",  # feet air time
        "clock_data",  # feet contact time
        "clock_data",  # foot error
]
obs_space_names_critic += ["heightmap:4x4"]


# Action Space
action_space_names = ["joints_pos"]


# Joints Order
joints_order = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
]


# Robot Name
robot_name = "a1"


morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = action_space_names,
        joints_order = joints_order,
        robot_name = robot_name,
    )

# add the heightmap to the obs space
vision_obs_space_names_actor = single_state_names * int(history_length)
heightmap_name = "heightmap:13x13"
vision_obs_space_names_actor += [heightmap_name]
vision_obs_space_names_actor += ["invariant_scalar"]
vision_obs_space_names_critic = single_state_names * int(history_length)
vision_obs_space_names_critic += [
        "clock_data", "clock_data", "clock_data",  # P gains
        "clock_data", "clock_data", "clock_data",  # D gains
]
vision_obs_space_names_critic += ["invariant_scalar", "invariant_scalar", "clock_data", "clock_data", "clock_data"]
vision_obs_space_names_critic += [heightmap_name]
vision_obs_space_names_critic += ["invariant_scalar"]

vision_morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = vision_obs_space_names_actor,
        obs_space_names_critic = vision_obs_space_names_critic,
        action_space_names = action_space_names,
        joints_order = joints_order,
        robot_name = robot_name,
    )

obs_state_ratio = 3
obs_space_names_critic += single_state_names * int(obs_state_ratio)

dae_morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        obs_space_names_single_state = single_state_names,
        action_space_names = action_space_names,
        joints_order = joints_order,
        robot_name = robot_name,
    )

# Include the heightmap in the single state representation for the DAE
vision_obs_space_names_critic += single_state_names * int(obs_state_ratio)

vision_dae_morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = vision_obs_space_names_actor,
        obs_space_names_critic = vision_obs_space_names_critic,
        obs_space_names_single_state = single_state_names,
        action_space_names = action_space_names,
        joints_order = joints_order,
        robot_name = robot_name,
    )