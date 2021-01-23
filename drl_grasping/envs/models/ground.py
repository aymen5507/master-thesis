from gym_ignition.scenario import model_wrapper
from gym_ignition.utils import misc
from gym_ignition.utils.scenario import get_unique_model_name
from scenario import core as scenario
from typing import List


class Ground(model_wrapper.ModelWrapper):

    def __init__(self,
                 world: scenario.World,
                 position: List[float] = (0, 0, 0),
                 orientation: List[float] = (1, 0, 0, 0),
                 size: List[float] = (1.0, 1.0),
                 collision_thickness=0.05):

        # Get a unique model name
        model_name = get_unique_model_name(world, 'ground')

        # Initial pose
        initial_pose = scenario.Pose(position, orientation)

        # Create SDF string for the model
        sdf = \
            f"""<sdf version="1.7">
            <model name="{model_name}">
                <static>true</static>
                <link name="{model_name}_link">
                    <collision name="{model_name}_collision">
                        <pose>0 0 {-collision_thickness/2} 0 0 0</pose>
                        <geometry>
                            <box>
                                <size>{size[0]} {size[1]} {collision_thickness}</size>
                            </box>
                        </geometry>
                    </collision>
                    <visual name="%s_visual">
                        <geometry>
                            <plane>
                                <normal>0 0 1</normal>
                                <size>{size[0]} {size[1]}</size>
                            </plane>
                        </geometry>
                        <material>
                            <ambient>0.8 0.8 0.8 1</ambient>
                            <diffuse>0.8 0.8 0.8 1</diffuse>
                            <specular>0.8 0.8 0.8 1</specular>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>"""

        # Convert it into a file
        sdf_file = misc.string_to_file(sdf)

        # Insert the model
        ok_model = world.to_gazebo().insert_model(sdf_file,
                                                  initial_pose,
                                                  model_name)
        if not ok_model:
            raise RuntimeError('Failed to insert ' + model_name)

        # Get the model
        model = world.get_model(model_name)

        # Initialize base class
        model_wrapper.ModelWrapper.__init__(self, model=model)
