# ==========================================
# DOFBOT MAIN PROGRAM
# ==========================================
#
# Description:
#     Coordinates command execution for the
#     Yahboom DOFBOT robotic arm.
#
#     Reads predefined commands from
#     dofbot_commands.py and generates joint-space
#     trajectories using dofbot_kinematics.py.
#
#     Executes movements and gripper operations
#     through dofbot_controller.py.
#
#     Optionally integrates with the 3D simulator
#     to display robot motion and record command
#     sequence history.
#
# ==========================================

# only needed if running with IPYNB files on Jupyter, comment out otherwise 
# import import_ipynb 

import time
import numpy as np
import dofbot_kinematics as kinematics
import dofbot_controller as controller
import constants as const


# ==========================================
# EXECUTE COMMAND SEQUENCE
# ==========================================

# Executes the active command sequence using the selected robot backend.
# Coordinates trajectory generation, gripper actions, and visualization.
def run_commands(visualizer=None):

    # Initial robot states
    gripper_state = controller.get_gripper_angle()

    joints_state = np.array(
        controller.get_joint_angles(),
        dtype=float
    )

    # Command sequence information
    sequence = const.ACTIVE_SEQUENCE

    total_commands = len(sequence)

    total_moves = sum(
        1 for command_type, _ in sequence
        if command_type == "MOVE_TO"
    )

    move_number = 0

    if visualizer is not None:
        visualizer.begin_sequence(total_commands)

    try:

        for command_number, (command_type, value) in enumerate(
            sequence,
            start=1
        ):

            # ==================================
            # VALIDATE COMMAND
            # ==================================

            if command_type not in ("MOVE_TO", "GRAB", "DROP"):

                print(
                    f"Unknown command type: {command_type}"
                )

                continue

            # Current joint configuration
            qstart = np.array(
                controller.get_joint_angles(),
                dtype=float
            )

            print(
                f"Executing {command_type} "
                f"({command_number}/{total_commands})"
            )

            print(
                f"Current joint positions: {qstart}"
            )

            # ==================================
            # MOVE ARM
            # ==================================

            if command_type == "MOVE_TO":

                move_number += 1

                qend = np.asarray(
                    value,
                    dtype=float
                ).flatten()

                if qend.size != 5:
                    raise ValueError(
                        "MOVE_TO requires 5 joint angles."
                    )

                print(
                    f"Desired joint positions: {qend}"
                )

                # Generate joint-space trajectory
                kinematics.generate_trajectory(
                    qstart,
                    qend
                )

                # Record movement in visualization
                if visualizer is not None:

                    visualizer.new_command(
                        command_type,
                        gripper_state,
                        command_number=command_number,
                        command_total=total_commands,
                        move_number=move_number,
                        move_total=total_moves
                    )

                # Execute joint-space trajectory
                controller.execute_trajectory(
                    gripper_state,
                    visualizer
                )

            # ==================================
            # GRIPPER COMMANDS
            # ==================================

            elif command_type in ("GRAB", "DROP"):

                # Record stationary gripper command
                if visualizer is not None:

                    visualizer.record_gripper_command(
                        command_type,
                        gripper_state,
                        qstart,
                        command_number=command_number,
                        command_total=total_commands,
                        move_total=total_moves
                    )

                # Command gripper only
                if command_type == "GRAB":
                    controller.command_gripper(
                        controller.GRIPPER_CLOSED,
                        visualizer=visualizer
                    )

                elif command_type == "DROP":
                    controller.command_gripper(
                        controller.GRIPPER_OPEN,
                        visualizer=visualizer
                    )

            # ==================================
            # UPDATE ROBOT STATE
            # ==================================

            joints_state = np.array(
                controller.get_joint_angles(),
                dtype=float
            )

            gripper_state = (
                controller.get_gripper_angle()
            )

            # Update gripper visualization
            if visualizer is not None:

                visualizer.set_gripper_angle(
                    gripper_state
                )

    finally:

        if visualizer is not None:
            visualizer.end_sequence()

    # ==========================================
    # FINISHED
    # ==========================================

    print(
        "Finished executing command sequence."
    )

    print(
        "Final joint positions:",
        joints_state
    )

    print(
        "Final gripper angle:",
        gripper_state
    )

    return joints_state


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    run_commands()
