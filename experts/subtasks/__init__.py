"""Per-subtask Markovian FSMs and the registry that maps names to instances.

Add a new subtask by writing its module (a `Subtask` subclass, see base.py) and
registering one instance in SUBTASKS below.
"""

from .base import (
    BONUS_THRESH,
    GripperCmd,
    HOME_EE_POS,
    HOME_EE_ROT,
    RESET_ARM_QPOS,
    SUBTASK_INFO,
    Subtask,
    at_home,
    is_done,
)
from .bottomknob import BottomknobSubtask
from .hinge import HingeSubtask
from .kettle import KettleSubtask
from .light import LightSubtask
from .microwave import MicrowaveSubtask
from .slide import SlideSubtask
from .topknob import TopknobSubtask

# Registry of implemented subtasks. Others are added incrementally.
SUBTASKS = {
    "microwave": MicrowaveSubtask(),
    "kettle": KettleSubtask(),
    "slide": SlideSubtask(),
    "light": LightSubtask(),
    "topknob": TopknobSubtask(),
    "bottomknob": BottomknobSubtask(),
    "hinge": HingeSubtask(),
}

__all__ = [
    "SUBTASKS",
    "Subtask",
    "GripperCmd",
    "HOME_EE_POS",
    "HOME_EE_ROT",
    "RESET_ARM_QPOS",
    "SUBTASK_INFO",
    "BONUS_THRESH",
    "is_done",
    "at_home",
    "MicrowaveSubtask",
    "KettleSubtask",
    "SlideSubtask",
    "LightSubtask",
    "TopknobSubtask",
    "BottomknobSubtask",
    "HingeSubtask",
]
