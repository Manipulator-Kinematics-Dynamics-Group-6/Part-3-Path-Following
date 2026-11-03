# ==========================================
# DOFBOT ROBOT CONTROLLER
# ==========================================
#
# Description:
#     Handles trajectory execution, joint commands,
#     gripper control, and robot state retrieval.
#
#     Supports both simulated and physical operation
#     through the TEST flag.
#
#     TEST = True:
#         Uses simulated joint and gripper states
#         without requiring the Arm_Lib library.
#
#     TEST = False:
#         Communicates with the physical DOFBOT
#         through the Arm_Lib interface.
#
#     Also manages trajectory timing and simulation
#     pause/resume functionality.
#
# ==========================================


import numpy as np
import time

import dofbot_kinematics as kinematics


# ==========================================
# CONTROLLER SETTINGS
# ==========================================

TEST = True

# Total time for each movement (seconds)
TOTAL_TIME = 10.0

# Gripper constants
GRIPPER_ID = 6

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130


# ==========================================
# SIMULATED ROBOT STATE
# ==========================================

simulated_q = np.array(
    [90, 90, 90, 90, 90],
    dtype=float
)

simulated_gripper = GRIPPER_OPEN


# ==========================================
# HARDWARE CONNECTION
# ==========================================

if not TEST:

    from Arm_Lib import Arm_Device

    arm = Arm_Device()


# ==========================================
# JOINT CONTROL
# ==========================================

# Sends desired joint angles to the simulated or physical robot.
def command_robot(joint_angles, gripper_angle, movement_time=2500):

    global simulated_q, simulated_gripper

    joint_angles = np.asarray(
        joint_angles,
        dtype=float
    ).flatten()

    if joint_angles.size != 5:
        raise ValueError(
            "Joint angles must contain exactly 5 values."
        )

    if TEST:

        # Update simulated joint positions
        simulated_q = joint_angles.copy()

        # Update simulated gripper position
        simulated_gripper = float(gripper_angle)

    else:

        # Send joint angles to physical DOFBOT
        arm.Arm_serial_servo_write6(
            *joint_angles,
            gripper_angle,
            movement_time
        )

# Reads the current joint angles from the simulation or hardware.
def get_joint_angles(desired=None):

    if TEST:

        return simulated_q.copy()

    else:

        angles = []

        for joint in range(1, 6):

            angle = arm.Arm_serial_servo_read(
                joint
            )

            if angle is None:
                raise RuntimeError(
                    f"Could not read servo {joint}."
                )

            angles.append(angle)

        return np.array(
            angles,
            dtype=float
        )


# ==========================================
# GRIPPER CONTROL
# ==========================================

# Opens or closes the gripper to a specified servo angle.
def command_gripper(angle, movement_time=1000):

    global simulated_gripper

    if not 0 <= angle <= 180:

        raise ValueError(
            "Gripper angle must be between 0 and 180."
        )

    if TEST:

        simulated_gripper = float(angle)

        if angle >= GRIPPER_CLOSED:

            print("Gripper closed")

        elif angle <= GRIPPER_OPEN:

            print("Gripper opened")

    else:

        arm.Arm_serial_servo_write(
            GRIPPER_ID,
            angle,
            movement_time
        )

# Returns the current gripper servo angle.
def get_gripper_angle():

    if TEST:

        return simulated_gripper

    else:

        angle = arm.Arm_serial_servo_read(
            GRIPPER_ID
        )

        if angle is None:
            raise RuntimeError(
                "Could not read gripper position."
            )

        return angle


# ==========================================
# EXECUTE TRAJECTORY
# ==========================================

# Executes the generated joint trajectory at fixed time intervals.
# Supports pausing and resuming through the visualizer.
def execute_trajectory(gripper_state, visualizer=None):

    # Time between trajectory samples
    dt = TOTAL_TIME / (kinematics.N - 1)

    # Servo movement duration in milliseconds
    movement_time = max(
        100,
        int(dt * 1000)
    )

    print("Starting DOFBOT trajectory...")
    print(f"Trajectory points: {kinematics.N}")
    print(f"Total movement time: {TOTAL_TIME} seconds")

    start_time = time.monotonic()

    # Total time spent paused
    paused_duration = 0.0

    try:

        for ii in range(kinematics.N):

            # ==================================
            # PROCESS VISUALIZER EVENTS
            # ==================================

            if visualizer is not None:

                visualizer.process_events()

                paused_duration += (
                    visualizer.wait_if_paused()
                )


            # ==================================
            # MAINTAIN TRAJECTORY TIMING
            # ==================================

            target_time = (
                start_time
                + ii * dt
                + paused_duration
            )

            while True:

                if visualizer is not None:

                    visualizer.process_events()

                    pause_time = (
                        visualizer.wait_if_paused()
                    )

                    if pause_time > 0:

                        paused_duration += pause_time

                        target_time += pause_time

                remaining = (
                    target_time
                    - time.monotonic()
                )

                if remaining <= 0:
                    break

                time.sleep(
                    min(remaining, 0.01)
                )


            # ==================================
            # COMMAND ROBOT
            # ==================================

            desired_q = kinematics.q[:, ii].copy()

            command_robot(
                desired_q,
                gripper_state,
                movement_time
            )


            # ==================================
            # UPDATE VISUALIZATION
            # ==================================

            if visualizer is not None:

                visualizer.update(ii)


            # ==================================
            # PRINT PROGRESS
            # ==================================

            print(
                f"Frame {ii + 1}/{kinematics.N} | "
                f"Time: {ii * dt:.2f}s"
            )

        print("Trajectory completed!")

    except KeyboardInterrupt:

        print(
            "\nTrajectory interrupted by user."
        )

        raise

    finally:

        print("Controller finished.")
        

if __name__ == "__main__":
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
    print(get_joint_angles())

    print("Initial gripper angle:")
    print(get_gripper_angle())


    # ==========================================
    # COMMAND ROBOT
    # ==========================================

    command_robot(
        qend,
        GRIPPER_OPEN
    )

    print("Updated joint angles:")
    print(get_joint_angles())


    # ==========================================
    # FORWARD KINEMATICS
    # ==========================================

    Rot, Pot = kinematics.fwdkin_Dofbot(get_joint_angles())

    print("End effector position:")
    print(Pot)

    print("End effector rotation matrix:")
    print(Rot)


    # ==========================================
    # GRIPPER TEST
    # ==========================================

    command_gripper(GRIPPER_CLOSED)

    print("Updated gripper angle:")
    print(get_gripper_angle())
