
import numpy as np

import dofbot_kinematics as robot
import dofbot_controller as controller
import dofbot_commands as command_data


# ==========================================
# EXECUTE COMMAND SEQUENCE
# ==========================================

def run_commands(visualizer=None):

    # Initial robot states
    gripper_state = controller.GRIPPER_OPEN

    # Get initial joint positions
    if controller.TEST:
        joints_state = np.array(
            [90, 90, 90, 90, 90],
            dtype=float
        )
    else:
        joints_state = np.array(
            controller.get_joint_angles(),
            dtype=float
        )

    # Execute command sequence
    for command_type, value in command_data.ACTIVE_SEQUENCE:

        print(
            f"Executing command: {command_type} "
            f"with value: {value}"
        )

        # Determine starting joint configuration
        if controller.TEST:
            q_start = joints_state.copy()
        else:
            q_start = np.array(
                controller.get_joint_angles(),
                dtype=float
            )

        print(f"Current joint positions: {q_start}")

        # Target joint configuration
        qend = np.array(value, dtype=float)

        # ==================================
        # GENERATE TRAJECTORY
        # ==================================

        robot.generate_trajectory(q_start, qend)

        # ==================================
        # INITIALIZE VISUALIZATION
        # ==================================

        if visualizer is not None:

            visualizer.new_command(
                command_type,
                gripper_state
            )

        # ==================================
        # EXECUTE COMMAND
        # ==================================

        match command_type:

            case "GRAB":

                controller.execute_trajectory(
                    gripper_state,
                    visualizer
                )

                controller.command_gripper(
                    controller.GRIPPER_CLOSED
                )

                gripper_state = controller.GRIPPER_CLOSED

            case "DROP":

                controller.execute_trajectory(
                    gripper_state,
                    visualizer
                )

                controller.command_gripper(
                    controller.GRIPPER_OPEN
                )

                gripper_state = controller.GRIPPER_OPEN

            case "MOVE_TO":

                controller.execute_trajectory(
                    gripper_state,
                    visualizer
                )

            case _:

                print(f"Unknown command type: {command_type}")
                continue

        # ==================================
        # UPDATE ROBOT STATE
        # ==================================

        if controller.TEST:
            joints_state = qend.copy()

        else:
            joints_state = np.array(
                controller.get_joint_angles(),
                dtype=float
            )

        # Update visualizer gripper
        if visualizer is not None:
            visualizer.set_gripper_angle(
                gripper_state
            )

    # ======================================
    # FINISHED
    # ======================================

    print(
        "Finished executing command sequence."
    )

    print(
        "Final joint positions:",
        joints_state
    )

    return joints_state


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    run_commands()
