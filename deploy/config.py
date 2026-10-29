from selectors import _PollLikeSelector
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'go2', 'b2', 'pegasus'
scene = 'random_boxes'  # flat, random_boxes, random_pyramids, perlin

# ----------------------------------------------------------------------------------------------------------------
if(robot == "go2"):
    Kp_walking = 20.0
    Kd_walking = 2.0

    Kp_stand_up_and_down = 50.
    Kd_stand_up_and_down = 5.

elif(robot == "b2"):
    Kp_walking = 100.
    Kd_walking = 5.

    Kp_stand_up_and_down = 100.
    Kd_stand_up_and_down = 5.

elif(robot == "pegasus"):
    Kp_walking = 200.
    Kd_walking = 10.

    Kp_stand_up_and_down = 200.
    Kd_stand_up_and_down = 10.

else:
    raise ValueError(f"Robot {robot} not supported")

# ----------------------------------------------------------------------------------------------------------------
# policy_folder_path = dir_path + "/../tested_policies/" + robot + "/moe"
# policy_folder_path = dir_path + "/../tested_policies/" + robot + "/rough_vision_symm_ecdae_direct/" + "2026-09-15_13-13-19_skoop_isaaclabv3"
policy_folder_path = dir_path + "/../tested_policies/" + robot + "/rough_vision_symm_erff_koopman_direct/" + "2026-10-27_18-42-18_erffkoop_new_noise_model_front_heightmap_only_no_lateral_no_backward_commands"
# policy_folder_path = dir_path + "/../tested_policies/" + robot + "/rough_vision_symm_direct/" + "2026-10-24_14-28-34_ppoeqic_new_noise_model"

# concurrent_state_est_network = policy_folder_path + "/exported/concurrent_state_estimator.pth"
concurrent_state_est_network = policy_folder_path + "/concurrent_state_estimator.pth"
rma_network = policy_folder_path + "/exported/rma.pth"

# Load specific training parameters
import yaml
with open(policy_folder_path + "/params/env.yaml", "r") as file:
    training_env = yaml.unsafe_load(file)
with open(policy_folder_path + "/params/agent.yaml", "r") as file:
    training_agent = yaml.unsafe_load(file)

use_vision = True  # If True, use the vision observations in the RL policy #TODO add in yaml
