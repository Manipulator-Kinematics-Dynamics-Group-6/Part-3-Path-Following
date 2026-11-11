import numpy as np

# ==========================================
# TRAJECTORY SETTINGS
# ==========================================

# N = 100
N = 20
lam = np.linspace(0, 1, N)


# ==========================================
# LINK LENGTHS (METERS)
# ==========================================

l0 = 0.061
l1 = 0.0435
l2 = 0.08285
l3 = 0.08285
l4 = 0.07385
l5 = 0.05457

LINK_LENGTHS = [l0, l1, l2, l3, l4, l5]


# ==========================================
# JOINT LIMITS (DEGREES)
# ==========================================

# Configure these to match the physical robot.
JOINT_MIN = np.array(
    [0, 0, 0, 0, 0],
    dtype=float
)

JOINT_MAX = np.array(
    [180, 180, 180, 180, 180],
    dtype=float
)


# ==========================================
# INVERSE KINEMATICS SETTINGS
# ==========================================

POSITION_TOL = 1e-4
ORIENTATION_TOL = 1e-3

IK_ATTEMPTS = 24



# ==========================================
# CONTROLLER SETTINGS
# ==========================================

# True if only simulating outputs 
TEST = False

# Total time for each movement (seconds)
TOTAL_TIME = 3

# Gripper constants
GRIPPER_ID = 6

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130

GRIPPER_WAIT = 1.5

# ==========================================
# SIMULATED ROBOT STATE
# ==========================================

simulated_q = np.array(
    [90, 90, 90, 90, 90],
    dtype=float
)

simulated_gripper = GRIPPER_OPEN

# ==========================================
# POSITION FEEDBACK SETTINGS
# ==========================================

GRIPPER_POSITION_TOL = 3.0       # Degrees
GRIPPER_POSITION_TIMEOUT = 5.0   # Seconds
GRIPPER_FEEDBACK_INTERVAL = 0.05

JOINT_POSITION_TOL = 3.0      # Degrees
JOINT_POSITION_TIMEOUT = 5.0  # Seconds
JOINT_FEEDBACK_INTERVAL = 0.05


# ==========================================
# COMMAND SEQUENCES
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

ACTIVE_SEQUENCE = RESET