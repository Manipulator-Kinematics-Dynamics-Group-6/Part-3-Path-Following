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

# only needed if running with IPYNB files on Jupyter, comment out otherwise 
# import import_ipynb  

import numpy as np
import time
import dofbot_kinematics as kinematics
import constants as const

# ==========================================
# HARDWARE CONNECTION
# ==========================================

if not const.TEST:
    from Arm_Lib import Arm_Device

    arm = Arm_Device()


# ==========================================
# JOINT CONTROL
# ==========================================

# Waits until all five joints reach their desired positions.
def wait_for_joints(target_angles, visualizer=None):

    target_angles = np.asarray(
        target_angles,
        dtype=float
    ).flatten()

    if target_angles.size != 5:
        raise ValueError(
            "Target must contain exactly 5 joint angles."
        )

    start_time = time.monotonic()

    while True:

        # ==================================
        # READ CURRENT JOINT POSITIONS
        # ==================================

        current_angles = np.asarray(
            get_joint_angles(),
            dtype=float
        )

        if current_angles.shape != (5,):
            raise RuntimeError(
                "Invalid joint feedback received."
            )

        if not np.all(np.isfinite(current_angles)):
            raise RuntimeError(
                "Joint feedback contains invalid values."
            )

        # ==================================
        # CALCULATE POSITION ERROR
        # ==================================

        error = np.abs(
            target_angles - current_angles
        )

        # Check all five joints
        reached = np.all(
            error <= const.JOINT_POSITION_TOL
        )

        if reached:

            print(
                "All joints reached their target positions."
            )

            print(
                f"Actual joint positions: {current_angles}"
            )

            print(
                f"Joint position errors: {error}"
            )

            return True

        # ==================================
        # CHECK TIMEOUT
        # ==================================

        elapsed = (
            time.monotonic() - start_time
        )

        if elapsed >= const.JOINT_POSITION_TIMEOUT:

            print(
                "Joint position timeout!"
            )

            print(
                f"Target positions: {target_angles}"
            )

            print(
                f"Actual positions: {current_angles}"
            )

            print(
                f"Position errors: {error}"
            )

            return False

        # ==================================
        # PROCESS VISUALIZATION EVENTS
        # ==================================

        if visualizer is not None:

            visualizer.process_events()

        time.sleep(
            const.JOINT_FEEDBACK_INTERVAL
        )


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

    if const.TEST:
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
def get_joint_angles():

    if const.TEST:
        return simulated_q.copy()

    else:
        time.sleep(.01)
        
        angles = []

        for joint in range(1, 6):

            angle = arm.Arm_serial_servo_read(
                joint
            )

            if angle is None:
                raise RuntimeError(
                    f"Could not read servo {joint}."
                )
#                 print("Could not read servo " + joint + ".")

            angles.append(angle)

        return np.array(
            angles,
            dtype=float
        )


# ==========================================
# GRIPPER CONTROL
# ==========================================

# Waits until the gripper reaches its target or times out.
def wait_for_gripper(target_angle, visualizer=None):

    start_time = time.monotonic()

    while True:

        current_angle = get_gripper_angle()

        error = abs(target_angle - current_angle)

        if error <= const.GRIPPER_POSITION_TOL:

            print(
                f"Gripper reached target: "
                f"{current_angle:.1f} deg"
            )

            return True

        if time.monotonic() - start_time >= const.GRIPPER_POSITION_TIMEOUT:

            print(
                f"Gripper timeout! "
                f"Target: {target_angle:.1f}, "
                f"Actual: {current_angle:.1f}"
            )

            return False

        if visualizer is not None:

            visualizer.process_events()

        time.sleep(const.GRIPPER_FEEDBACK_INTERVAL)

# Commands the gripper and waits for position feedback.
def command_gripper(angle, movement_time=1000, visualizer=None):

    global simulated_gripper

    if not 0 <= angle <= 180:

        raise ValueError(
            "Gripper angle must be between 0 and 180."
        )

    if const.TEST:

        simulated_gripper = float(angle)

    else:

        arm.Arm_serial_servo_write(
            const.GRIPPER_ID,
            angle,
            movement_time
        )

    # Wait for the target position
    reached = wait_for_gripper(
        angle,
        visualizer
    )

    if not reached:

        raise TimeoutError(
            "Gripper failed to reach its target."
        )

# Returns the current gripper servo angle.
def get_gripper_angle():

    if const.TEST:
        return simulated_gripper

    else:
        time.sleep(.01)
        
        angle = arm.Arm_serial_servo_read(
            const.GRIPPER_ID
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
    dt = const.TOTAL_TIME / (kinematics.N - 1)

    # Servo movement duration in milliseconds
    movement_time = max(
        100,
        int(dt * 1000)
    )

    print("Starting DOFBOT trajectory...")
    print(f"Trajectory points: {kinematics.N}")
    print(f"Total movement time: {const.TOTAL_TIME} seconds")

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
                f"Desired Angles: {kinematics.q[:, ii]} | "
                f"Actual Angles: {get_joint_angles()} | "
                f"Time: {ii * dt:.2f}s"
            )

        # ==================================
        # VERIFY FINAL JOINT POSITIONS
        # ==================================

        # Get final desired configuration
        target_angles = kinematics.q[:, -1].copy()

        print(
            "Trajectory commands completed. "
            "Verifying final joint positions..."
        )

        reached = wait_for_joints(
            target_angles,
            visualizer
        )

        if not reached:

            raise TimeoutError(
                "Robot failed to reach the final "
                "joint configuration."
            )

        print(
            "Trajectory completed successfully!"
        )
        
    except KeyboardInterrupt:
        print("\nTrajectory interrupted by user.")
        raise

    finally:
        print("Controller finished.")

    
# ==========================================
# TEST CONFIGURATIONS
# ==========================================
    
def test_config() :

    qstart = np.array(
        [90, 90, 90, 90, 90],
        dtype=float
    )

    qend = np.array(
        [45, 120, 140, 100, 95],
        dtype=float
    )

    print("\n==========================================")
    print("DOFBOT CONTROLLER TEST")
    print("==========================================")

    # Prevent accidental hardware movement
    assert const.TEST, (
        "Controller tests require TEST = True."
    )

    # ==========================================
    # RESET SIMULATED ROBOT STATE
    # ==========================================

    simulated_q = qstart.copy()
    simulated_gripper = const.GRIPPER_OPEN

    # ==========================================
    # TEST 1 - INITIAL ROBOT STATE
    # ==========================================

    print("\nTEST 1: Initial robot state")

    actual_q = get_joint_angles()
    actual_gripper = get_gripper_angle()

    assert np.allclose(
        actual_q,
        qstart
    ), (
        f"Initial joint positions incorrect: {actual_q}"
    )

    assert np.isclose(
        actual_gripper,
        const.GRIPPER_OPEN
    ), (
        f"Initial gripper angle incorrect: {actual_gripper}"
    )

    print("PASS: Initial robot state")


    # ==========================================
    # TEST 2 - JOINT COMMAND
    # ==========================================

    print("\nTEST 2: Joint command")

    command_robot(
        qend,
        const.GRIPPER_OPEN
    )

    actual_q = get_joint_angles()

    assert np.allclose(
        actual_q,
        qend
    ), (
        f"Joint command failed. "
        f"Expected: {qend}, Actual: {actual_q}"
    )

    print("PASS: Joint command")


    # ==========================================
    # TEST 3 - JOINT POSITION FEEDBACK
    # ==========================================

    print("\nTEST 3: Joint position feedback")

    reached = wait_for_joints(
        qend
    )

    assert reached, (
        "Joint position verification failed."
    )

    print("PASS: Joint position feedback")


    # ==========================================
    # TEST 4 - TRAJECTORY GENERATION
    # ==========================================

    print("\nTEST 4: Trajectory generation")

    kinematics.generate_trajectory(
        qstart,
        qend
    )

    assert kinematics.q.shape == (
        5,
        kinematics.N
    ), (
        "Generated trajectory has incorrect dimensions."
    )

    assert np.allclose(
        kinematics.q[:, 0],
        qstart
    ), (
        "Trajectory does not start at qstart."
    )

    assert np.allclose(
        kinematics.q[:, -1],
        qend
    ), (
        "Trajectory does not end at qend."
    )

    assert np.all(
        np.isfinite(kinematics.q)
    ), (
        "Trajectory contains invalid numerical values."
    )

    print("PASS: Trajectory generation")


    # ==========================================
    # TEST 5 - TRAJECTORY EXECUTION
    # ==========================================

    print("\nTEST 5: Trajectory execution")

    # Reset simulated robot to initial configuration
    simulated_q = qstart.copy()

    execute_trajectory(
        const.GRIPPER_OPEN
    )

    actual_q = get_joint_angles()

    joint_error = np.abs(
        qend - actual_q
    )

    assert np.all(
        joint_error <= const.JOINT_POSITION_TOL
    ), (
        f"Trajectory execution failed. "
        f"Joint errors: {joint_error}"
    )

    print("PASS: Trajectory execution")


    # ==========================================
    # TEST 6 - FORWARD KINEMATICS
    # ==========================================

    print("\nTEST 6: Forward kinematics")

    Rot, Pot = kinematics.fwdkin_Dofbot(
        actual_q
    )

    assert Rot.shape == (3, 3), (
        "Rotation matrix has incorrect dimensions."
    )

    assert Pot.shape == (3,), (
        "End-effector position has incorrect dimensions."
    )

    assert np.allclose(
        Rot.T @ Rot,
        np.eye(3),
        atol=1e-6
    ), (
        "Rotation matrix is not orthogonal."
    )

    assert np.isclose(
        np.linalg.det(Rot),
        1.0,
        atol=1e-6
    ), (
        "Rotation matrix has invalid determinant."
    )

    assert np.all(
        np.isfinite(Pot)
    ), (
        "End-effector position contains invalid values."
    )

    print("PASS: Forward kinematics")


    # ==========================================
    # TEST 7 - GRIPPER CLOSE
    # ==========================================

    print("\nTEST 7: Gripper close")

    command_gripper(
        const.GRIPPER_CLOSED
    )

    actual_gripper = get_gripper_angle()

    assert abs(
        const.GRIPPER_CLOSED - actual_gripper
    ) <= const.GRIPPER_POSITION_TOL, (
        f"Gripper failed to close. "
        f"Expected: {const.GRIPPER_CLOSED}, "
        f"Actual: {actual_gripper}"
    )

    print("PASS: Gripper close")


    # ==========================================
    # TEST 8 - GRIPPER OPEN
    # ==========================================

    print("\nTEST 8: Gripper open")

    command_gripper(
        const.GRIPPER_OPEN
    )

    actual_gripper = get_gripper_angle()

    assert abs(
        const.GRIPPER_OPEN - actual_gripper
    ) <= const.GRIPPER_POSITION_TOL, (
        f"Gripper failed to open. "
        f"Expected: {const.GRIPPER_OPEN}, "
        f"Actual: {actual_gripper}"
    )

    print("PASS: Gripper open")


    # ==========================================
    # TEST COMPLETE
    # ==========================================

    print("\n==========================================")
    print("ALL CONTROLLER TESTS PASSED")
    print("==========================================")


# ==========================================
# MAIN - CONTROLLER TEST
# ==========================================

if __name__ == "__main__":
    test_config()
