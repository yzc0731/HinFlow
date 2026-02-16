import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from .utils import xml_path_completion


class OnTheGroundKinova3(ManipulatorModel):
    """
    The Gen3 robot is the sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "osc_pose"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.333, 0.000, 1.750, 0.000, 1.040, -np.pi / 2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0.42,
            ),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
