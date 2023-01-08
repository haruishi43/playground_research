import os
import shutil
import numpy as np
import magnum as mn

import habitat
import habitat_sim
import habitat_sim.utils

# path
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from actions import NewMove

# vis
from pprint import pprint
import cv2
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

### PARAMS
# body movement
FORWARD_KEY="w"
BACKWARD_KEY="s"
LEFT_KEY="a"
RIGHT_KEY="d"
# camera movemnt
L_UP_KEY="u"
L_DOWN_KEY="j"
L_LEFT_KEY="h"
L_RIGHT_KEY="k"
NO_MOVE_KEY="i"
# terminal
FINISH="f"
# paths
IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)



def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def agent_info(sim):
    print(sim.is_episode_active())

    pass


def example(env):

    # follower
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env.sim, goal_radius, False)
    follower.mode = 'geodesic_path'

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0], observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    # episode
    goal_pos = env.current_episode.goals[0].position

    count_steps = 0
    while not env.episode_over:
        
        # step body
        best_action = follower.get_next_action(goal_pos)
        print(f"body action: {best_action}")
        body_observations = env.step(best_action)

        state = env.sim.get_agent_state(0)
        print(state.position)

        if env.episode_over:
            print('episode over!')
            break

        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = habitat.SimulatorActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(BACKWARD_KEY):
            action = habitat.SimulatorActions.MOVE_BACKWARD
            print("action: BACKWARD")
        elif keystroke == ord(LEFT_KEY):
            action = habitat.SimulatorActions.MOVE_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = habitat.SimulatorActions.MOVE_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(L_LEFT_KEY):
            action = habitat.SimulatorActions.LOOK_LEFT
            print("action: LOOK LEFT")
        elif keystroke == ord(L_RIGHT_KEY):
            action = habitat.SimulatorActions.LOOK_RIGHT
            print("action: LOOK RIGHT")
        elif keystroke == ord(L_UP_KEY):
            action = habitat.SimulatorActions.LOOK_UP
            print("action: LOOK UP")
        elif keystroke == ord(L_DOWN_KEY):
            action = habitat.SimulatorActions.LOOK_DOWN
            print("action: NO MOVE")
        elif keystroke == ord(NO_MOVE_KEY):
            action = habitat.SimulatorActions.NO_MOVE
            print("action: LOOK DOWN")
        elif keystroke == ord(FINISH):
            action = habitat.SimulatorActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0], observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    print(observations["pointgoal_with_gps_compass"][0])
    if observations["pointgoal_with_gps_compass"][0] < 0.25:  #action == habitat.SimulatorActions.STOP and observations["pointgoal"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")



if __name__ == "__main__":
    
    habitat.SimulatorActions.extend_action_space("MOVE_BACKWARD")
    habitat.SimulatorActions.extend_action_space("MOVE_LEFT")
    habitat.SimulatorActions.extend_action_space("MOVE_RIGHT")
    habitat.SimulatorActions.extend_action_space("LOOK_RIGHT")
    habitat.SimulatorActions.extend_action_space("LOOK_LEFT")
    habitat.SimulatorActions.extend_action_space("NO_MOVE")

    config = habitat.get_config("configs/tasks/pointnav2.yaml")

    config.defrost()
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NewMove"
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    # print(len(habitat.SimulatorActions))
    # pprint(habitat.SimulatorActions)
    # pprint(list(zip(enumerate(iter(habitat.SimulatorActions)))))
    # a = habitat.SimulatorActions.STOP
    # print(a)

    pprint(config)

    env = habitat.Env(config=config)
    example(env)