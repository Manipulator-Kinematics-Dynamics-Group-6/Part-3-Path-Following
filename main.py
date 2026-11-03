
import numpy as np

import dofbot_kinematics as robot
import dofbot_controller as controller
import dofbot_commands as command_data


# ==========================================
# EXECUTE COMMAND SEQUENCE
# ==========================================

def run_commands(visualizer=None):

    gripper_state = controller.get_gripper_angle()
    joints_state = np.array(controller.get_joint_angles(), dtype=float)

    sequence = command_data.ACTIVE_SEQUENCE
    total_commands = len(sequence)
    total_moves = sum(1 for kind, _ in sequence if kind == "MOVE_TO")
    move_number = 0

    if visualizer is not None:
        visualizer.begin_sequence(total_commands)

    try:
        for command_number, (command_type, value) in enumerate(sequence, start=1):

            if command_type not in ("MOVE_TO", "GRAB", "DROP"):
                print(f"Unknown command type: {command_type}")
                continue

            if command_type == "MOVE_TO":
                move_number += 1

            qstart = np.array(controller.get_joint_angles(), dtype=float)
            qend = np.array(value, dtype=float)

            print(f"Executing {command_type} ({command_number}/{total_commands})")
            print(f"Original joint positions: {qstart}")
            print(f"Desired joint positions: {qend}")

            robot.generate_trajectory(qstart, qend)

            if visualizer is not None:
                visualizer.new_command(
                    command_type,
                    gripper_state,
                    command_number=command_number,
                    command_total=total_commands,
                    move_number=move_number if command_type == "MOVE_TO" else None,
                    move_total=total_moves
                )

            controller.execute_trajectory(gripper_state, visualizer)

            if command_type == "GRAB":
                controller.command_gripper(controller.GRIPPER_CLOSED)
            elif command_type == "DROP":
                controller.command_gripper(controller.GRIPPER_OPEN)

            gripper_state = controller.get_gripper_angle()
            joints_state = np.array(controller.get_joint_angles(), dtype=float)

            if visualizer is not None:
                visualizer.set_gripper_angle(gripper_state)

    finally:
        if visualizer is not None:
            visualizer.end_sequence()

    print("Finished executing command sequence.")
    print("Final joint positions:", joints_state)
    return joints_state


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    run_commands()
