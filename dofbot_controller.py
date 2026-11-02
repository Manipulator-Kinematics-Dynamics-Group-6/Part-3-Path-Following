
import numpy as np
import time

import dofbot_kinematics as robot

# ==========================================
# HARDWARE CONNECTION
# ==========================================

from Arm_Lib import Arm_Device

arm = Arm_Device()


# ==========================================
# CONTROLLER SETTINGS
# ==========================================

# IMPORTANT:
# Keep True until you have checked the servo
# angle mapping and tested the robot safely.

DRY_RUN = True

# Total time for the movement (seconds)
TOTAL_TIME = 10.0

# Servo IDs for the five robot joints
SERVO_IDS = [1, 2, 3, 4, 5]

# Mapping between mathematical joint angles
# and physical servo angles.
#
# These are INITIAL PLACEHOLDERS.
# You must calibrate offsets and directions
# against your physical robot.

SERVO_OFFSETS = [0, 0, 0, 0, 0]
SERVO_DIRECTIONS = [1, 1, 1, 1, 1]

# Gripper servo
GRIPPER_ID = 6

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130


# ==========================================
# CONVERT JOINT ANGLES TO SERVO ANGLES
# ==========================================

def convert_to_servo_angles(joint_angles):

    servo_angles = []

    for i in range(5):

        angle = (
            SERVO_OFFSETS[i]
            + SERVO_DIRECTIONS[i] * joint_angles[i]
        )

        # Check servo limits
        if not 0 <= angle <= 180:
            raise ValueError(
                f"Servo {SERVO_IDS[i]} angle "
                f"{angle} is outside 0-180 degrees."
            )

        servo_angles.append(int(round(angle)))

    return servo_angles


# ==========================================
# SEND JOINT COMMANDS
# ==========================================

def command_robot(joint_angles, movement_time):

    servo_angles = convert_to_servo_angles(
        joint_angles
    )

    if DRY_RUN:

        print("Desired joint angles:", joint_angles)
        print("Servo commands:", servo_angles)

    else:

        for servo_id, angle in zip(
            SERVO_IDS, servo_angles
        ):

            arm.Arm_serial_servo_write(
                servo_id,
                angle,
                movement_time
            )


# ==========================================
# GRIPPER CONTROL
# ==========================================

def command_gripper(angle, movement_time=1500):

    if not 0 <= angle <= 180:
        raise ValueError("Gripper angle must be 0-180.")

    if DRY_RUN:
        print("Gripper command:", angle)

    else:
        arm.Arm_serial_servo_write(
            GRIPPER_ID,
            angle,
            movement_time
        )


# ==========================================
# EXECUTE TRAJECTORY
# ==========================================

def execute_trajectory():

    # Time between trajectory samples
    dt = TOTAL_TIME / (robot.N - 1)

    # Servo movement duration in milliseconds
    movement_time = max(100, int(dt * 1000))

    print("Starting Dofbot trajectory...")
    print(f"Trajectory points: {robot.N}")
    print(f"Total movement time: {TOTAL_TIME} seconds")

    # Start with gripper open
    command_gripper(GRIPPER_OPEN)

    start_time = time.monotonic()

    try:
        for ii in range(robot.N):

            # Desired joint configuration
            desired_q = robot.q[:, ii]

            # Send commands to the robot
            command_robot(desired_q, movement_time)

            # Maintain trajectory timing
            target_time = start_time + ii * dt

            remaining = target_time - time.monotonic()

            if remaining > 0:
                time.sleep(remaining)

            print(
                f"Frame {ii+1}/{robot.N} | "
                f"Time: {ii*dt:.2f}s"
            )

        print("Trajectory completed!")

    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user.")

    finally:
        print("Controller finished.")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    execute_trajectory()
