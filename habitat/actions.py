
import numpy as np
import magnum as mn

import habitat
from habitat.sims.habitat_simulator.action_spaces import (
    HabitatSimV1ActionSpaceConfiguration,
    HabitatSimV0ActionSpaceConfiguration,
    ActionSpaceConfiguration,
)
import habitat_sim
from habitat_sim.agent.controls import register_move_fn


X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

rotate_local_fns = [
    habitat_sim.SceneNode.rotate_x_local,
    habitat_sim.SceneNode.rotate_y_local,
    habitat_sim.SceneNode.rotate_z_local,    
]

rotate_global_fns = [
    habitat_sim.SceneNode.rotate_x,
    habitat_sim.SceneNode.rotate_y,
    habitat_sim.SceneNode.rotate_z,    
]


def rotate_local(scene_node: habitat_sim.SceneNode, theta: float, axis: int):
    rotate_local_fns[axis](scene_node, mn.Deg(theta))
    scene_node.rotation = scene_node.rotation.normalized()

def rotate_global(scene_node: habitat_sim.SceneNode, theta: float, axis: int):
    rotate_global_fns[axis](scene_node, mn.Deg(theta))
    scene_node.rotation = scene_node.rotation.normalized()

@register_move_fn(body_action=False)
class YAxisRot(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: habitat_sim.ActuationSpec):
        rotate_global(scene_node, actuation_spec.amount, Y_AXIS)


@register_move_fn(body_action=False)
class XAxisRot(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: habitat_sim.ActuationSpec):
        rotate_local(scene_node, actuation_spec.amount, X_AXIS)


@habitat.registry.register_action_space_configuration
class NewMove(HabitatSimV0ActionSpaceConfiguration):
    def get(self):
        config = super().get()
        
        # Body movement:
        config[habitat.SimulatorActions.MOVE_BACKWARD] = habitat_sim.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25))
        config[habitat.SimulatorActions.MOVE_LEFT] = habitat_sim.ActionSpec(
            "move_left", habitat_sim.ActuationSpec(amount=0.25))
        config[habitat.SimulatorActions.MOVE_RIGHT] = habitat_sim.ActionSpec(
            "move_right", habitat_sim.ActuationSpec(amount=0.25))
        
        # Head rotation:
        config[habitat.SimulatorActions.LOOK_DOWN] = habitat_sim.ActionSpec(
            "x_axis_rot", habitat_sim.agent.ActuationSpec(amount=-5.0)) 
        config[habitat.SimulatorActions.LOOK_LEFT] = habitat_sim.ActionSpec(
            "y_axis_rot", habitat_sim.agent.ActuationSpec(amount=5.0))
        config[habitat.SimulatorActions.LOOK_RIGHT] = habitat_sim.ActionSpec(
            "y_axis_rot", habitat_sim.agent.ActuationSpec(amount=-5.0))
        config[habitat.SimulatorActions.LOOK_UP] = habitat_sim.ActionSpec(
            "x_axis_rot", habitat_sim.agent.ActuationSpec(amount=5.0))
        config[habitat.SimulatorActions.NO_MOVE] = habitat_sim.ActionSpec(
            "x_axis_rot", habitat_sim.agent.ActuationSpec(amount=0.0)) 

        return config
