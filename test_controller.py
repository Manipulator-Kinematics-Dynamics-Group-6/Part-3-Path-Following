import numpy as np

import dofbot_controller as controller
import dofbot_kinematics as robot


# ==========================================
# TEST CONFIGURATIONS
# ==========================================

qstart = np.array(
    [90, 90, 90, 90, 90],
    dtype=float
)

qend = np.array(
    [45, 120, 140, 100, 95],
    dtype=float
)


# ==========================================
# INITIAL ROBOT STATE
# ==========================================

print("Initial joint angles:")
print(controller.get_joint_angles())

print("Initial gripper angle:")
print(controller.get_gripper_angle())


# ==========================================
# COMMAND ROBOT
# ==========================================

controller.command_robot(
    qend,
    controller.GRIPPER_OPEN
)

print("Updated joint angles:")
print(controller.get_joint_angles())


# ==========================================
# FORWARD KINEMATICS
# ==========================================

Rot, Pot = robot.fwdkin_Dofbot(
    controller.get_joint_angles()
)

print("End effector position:")
print(Pot)

print("End effector rotation matrix:")
print(Rot)


# ==========================================
# GRIPPER TEST
# ==========================================

controller.command_gripper(
    controller.GRIPPER_CLOSED
)

print("Updated gripper angle:")
print(controller.get_gripper_angle())