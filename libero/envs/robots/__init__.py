from .mounted_panda import MountedPanda
from .mounted_ur5e import MountedUR5e
from .mounted_jaco import MountedJaco
from .mounted_kinova3 import MountedKinova3
from .on_the_ground_panda import OnTheGroundPanda
from .on_the_ground_ur5e import OnTheGroundUR5e
from .on_the_ground_jaco import OnTheGroundJaco
from .on_the_ground_kinova3 import OnTheGroundKinova3

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "OnTheGroundUR5e": SingleArm,
        "MountedUR5e": SingleArm,
        "MountedJaco": SingleArm,
        "OnTheGroundJaco": SingleArm,
        "MountedKinova3": SingleArm,
        "OnTheGroundKinova3": SingleArm,
    }
)
