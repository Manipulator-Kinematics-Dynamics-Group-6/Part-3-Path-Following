import numpy as np
import time

import dofbot_kinematics as robot

# ==========================================
# CONTROLLER SETTINGS
# ==========================================

TEST = True

# Total time for the movement (seconds)
TOTAL_TIME = 10.0

# Gripper constants
GRIPPER_ID = 6

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130

# ==========================================
# HARDWARE CONNECTION
# ==========================================

if not TEST:
    from Arm_Lib import Arm_Device
    arm = Arm_Device()


# ==========================================
# JOINT CONTROL
# ==========================================

def command_robot(joint_angles, gripper_angle, movement_time=2500):
    if TEST:
        print("Desired joint angles:", joint_angles)

    else:
        arm.Arm_serial_servo_write6(
            joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], 
            gripper_angle, 
            movement_time)
        
def get_joint_angles(desired=None):
    if TEST:
        print("Getting joint angles in test mode.")
        return desired
    
    else:
        angles = []
        for joint in range(5):
            angles.append(arm.Arm_serial_servo_read(joint))
        return angles


# ==========================================
# GRIPPER CONTROL
# ==========================================

def command_gripper(angle, movement_time=1000):

    if not 0 <= angle <= 180:
        raise ValueError("Gripper angle must be 0-180.")

    if TEST:
        if (angle >= GRIPPER_CLOSED):
            print("Gripper closed")
        elif (angle <= GRIPPER_OPEN):
            print("Gripper opened")

    else:
        arm.Arm_serial_servo_write(GRIPPER_ID, angle, movement_time)


# ==========================================
# EXECUTE TRAJECTORY
# ==========================================

def execute_trajectory(gripper_state, visualizer=None):

    # Time between trajectory samples
    dt = TOTAL_TIME / (robot.N - 1)

    # Servo movement duration in milliseconds
    movement_time = max(100, int(dt * 1000))

    print("Starting Dofbot trajectory...")
    print(f"Trajectory points: {robot.N}")
    print(f"Total movement time: {TOTAL_TIME} seconds")

    start_time = time.monotonic()

    try:
        for ii in range(robot.N):

            # Desired joint configuration
            desired_q = robot.q[:, ii]

            # Send commands to the robot
            command_robot(desired_q, gripper_state, movement_time)

            # Maintain trajectory timing
            target_time = start_time + ii * dt

            remaining = target_time - time.monotonic()

            if remaining > 0:
                time.sleep(remaining)

            print(
                f"Frame {ii+1}/{robot.N} | "
                f"Time: {ii*dt:.2f}s"
            )
            
            if visualizer is not None:
                visualizer.update(ii)

        print("Trajectory completed!")

    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user.")

    finally:
        print("Controller finished.")
