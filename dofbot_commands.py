# ==========================================
# DOFBOT COMMAND SEQUENCES
# ==========================================
#
# Description:
#     Defines predefined command sequences for the
#     Yahboom DOFBOT robotic arm.
#
#     Commands include joint-space movements (MOVE_TO)
#     and gripper operations (GRAB and DROP).
#
#     The active command sequence is selected using
#     ACTIVE_SEQUENCE and executed by main.py.
#
# ==========================================


PICK_AND_PLACE = [
    ("MOVE_TO", [45, 120, 140, 100, 95]),
    ("GRAB", None),
    ("MOVE_TO", [90, 90, 90, 90, 90]),
    ("DROP", None)
]

TEST_SEQUENCE = [
    ("MOVE_TO", [45, 120, 140, 100, 95]),
    ("MOVE_TO", [90, 90, 90, 90, 90])
]

RESET = [
    ("MOVE_TO", [90, 90, 90, 90, 90]),
    ("DROP", None)
]

RESET_OTHER = [
    ("MOVE_TO", [45, 120, 140, 100, 95])
    ("GRAB", None)
]


# ==========================================
# SELECT ACTIVE COMMAND SEQUENCE
# ==========================================

ACTIVE_SEQUENCE = RESET
