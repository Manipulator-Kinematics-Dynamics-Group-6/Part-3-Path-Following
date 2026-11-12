
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

import dofbot_kinematics as kinematics


# ==========================================
# TRAJECTORY TRACKING
# ==========================================

class TrajectoryRecorder:

    # Initializes the desired and actual trajectory history.
    def __init__(self):

        self.start_time = time.monotonic()

        self.samples = []
        self.command_boundaries = []


    # Records the start of a new MOVE_TO command.
    def begin_command(self, command_number):

        elapsed = time.monotonic() - self.start_time

        self.command_boundaries.append(
            (elapsed, command_number)
        )


    # Records joint angles and calculated end-effector poses.
    def record(self, desired_q, actual_q):

        desired_q = np.asarray(
            desired_q, dtype=float
        ).reshape(5)

        actual_q = np.asarray(
            actual_q, dtype=float
        ).reshape(5)

        if not (
            np.all(np.isfinite(desired_q))
            and np.all(np.isfinite(actual_q))
        ):
            raise ValueError(
                "Invalid joint feedback received."
            )

        Rot_desired, Pot_desired = (
            kinematics.fwdkin_Dofbot(desired_q)
        )

        Rot_actual, Pot_actual = (
            kinematics.fwdkin_Dofbot(actual_q)
        )

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected"
            )

            rpy_desired = Rotation.from_matrix(
                Rot_desired
            ).as_euler("ZYX", degrees=True)[::-1]

            rpy_actual = Rotation.from_matrix(
                Rot_actual
            ).as_euler("ZYX", degrees=True)[::-1]

        orientation_error = np.rad2deg(
            Rotation.from_matrix(
                Rot_desired @ Rot_actual.T
            ).magnitude()
        )

        self.samples.append({
            "time": time.monotonic() - self.start_time,
            "desired_q": desired_q.copy(),
            "actual_q": actual_q.copy(),
            "desired_p": Pot_desired.copy(),
            "actual_p": Pot_actual.copy(),
            "desired_rpy": rpy_desired.copy(),
            "actual_rpy": rpy_actual.copy(),
            "orientation_error": orientation_error
        })


    # Creates comparison graphs for each vector component.
    def plot_components(
        self, t, desired, actual,
        labels, title, ylabel
    ):

        fig, axes = plt.subplots(
            len(labels),
            1,
            figsize=(11, 2 * len(labels) + 1),
            sharex=True
        )

        fig.suptitle(title)

        for ii, label in enumerate(labels):

            axes[ii].plot(
                t,
                desired[:, ii],
                "--",
                label=f"{label} desired"
            )

            axes[ii].plot(
                t,
                actual[:, ii],
                "-",
                label=f"{label} actual"
            )

            axes[ii].set_ylabel(ylabel)
            axes[ii].legend(loc="best")
            axes[ii].grid(True, alpha=0.3)

            self.mark_commands(axes[ii])

        axes[-1].set_xlabel("Elapsed time (s)")

        fig.tight_layout()


    # Marks the beginning of each MOVE_TO command.
    def mark_commands(self, ax):

        for elapsed, number in self.command_boundaries:

            ax.axvline(
                elapsed,
                color="gray",
                linewidth=0.8,
                alpha=0.5
            )


    # Displays the joint, EE pose, and tracking-error graphs.
    def plot(self):

        if len(self.samples) == 0:

            print("No trajectory tracking data recorded.")
            return

        t = np.array([
            sample["time"]
            for sample in self.samples
        ])

        q_desired = np.array([
            sample["desired_q"]
            for sample in self.samples
        ])

        q_actual = np.array([
            sample["actual_q"]
            for sample in self.samples
        ])

        p_desired = np.array([
            sample["desired_p"]
            for sample in self.samples
        ])

        p_actual = np.array([
            sample["actual_p"]
            for sample in self.samples
        ])

        rpy_desired = np.array([
            sample["desired_rpy"]
            for sample in self.samples
        ])

        rpy_actual = np.array([
            sample["actual_rpy"]
            for sample in self.samples
        ])

        # Unwrap angular data for continuous plots.
        rpy_desired = np.rad2deg(
            np.unwrap(
                np.deg2rad(rpy_desired),
                axis=0
            )
        )

        rpy_actual = np.rad2deg(
            np.unwrap(
                np.deg2rad(rpy_actual),
                axis=0
            )
        )

        # ==================================
        # JOINT ANGLE GRAPHS
        # ==================================

        self.plot_components(
            t,
            q_desired,
            q_actual,
            ["J1", "J2", "J3", "J4", "J5"],
            "Joint Angles: Desired vs Actual",
            "Angle (deg)"
        )

        # ==================================
        # END EFFECTOR POSITION GRAPHS
        # ==================================

        self.plot_components(
            t,
            p_desired * 1000,
            p_actual * 1000,
            ["X", "Y", "Z"],
            "End-Effector Position: Desired vs Actual",
            "Position (mm)"
        )

        # ==================================
        # END EFFECTOR ORIENTATION GRAPHS
        # ==================================

        self.plot_components(
            t,
            rpy_desired,
            rpy_actual,
            ["Roll", "Pitch", "Yaw"],
            "End-Effector Orientation: Desired vs Actual",
            "Angle (deg)"
        )

        # ==================================
        # TRACKING ERROR GRAPHS
        # ==================================

        joint_error = np.max(
            np.abs(q_desired - q_actual),
            axis=1
        )

        position_error = np.linalg.norm(
            p_desired - p_actual,
            axis=1
        ) * 1000

        orientation_error = np.array([
            sample["orientation_error"]
            for sample in self.samples
        ])

        fig, axes = plt.subplots(
            3, 1,
            figsize=(11, 8),
            sharex=True
        )

        fig.suptitle("Trajectory Tracking Errors")

        error_data = [
            (joint_error, "Maximum joint error (deg)"),
            (position_error, "EE position error (mm)"),
            (orientation_error, "EE orientation error (deg)")
        ]

        for ax, (values, label) in zip(axes, error_data):

            ax.plot(t, values)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

            self.mark_commands(ax)

        axes[-1].set_xlabel("Elapsed time (s)")

        fig.tight_layout()
