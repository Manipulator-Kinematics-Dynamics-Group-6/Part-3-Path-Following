# ==========================================
# DOFBOT ROBOT SIMULATION AND VISUALIZATION
# ==========================================
#
# Description:
#     Provides an interactive 3D visualization of
#     the Yahboom DOFBOT robotic arm using Matplotlib.
#
#     Displays joint configurations, end-effector
#     position and orientation, gripper state,
#     and trajectory execution information.
#
#     Records command-sequence history for playback,
#     frame navigation, and trajectory inspection.
#
#     Includes pause/resume, replay, camera reset,
#     and command navigation controls.
#
#     The visualization operates independently of
#     physical robot hardware.
#
# ==========================================


import bisect
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation

import dofbot_kinematics as kinematics


# ==========================================
# SIMULATOR SETTINGS
# ==========================================

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130

FINGER_LENGTH = 0.045
MAX_GAP = 0.045
MIN_GAP = 0.008

TOTAL_TIME = 10.0
MAX_FPS = 20.0


# ==========================================
# VISUALIZER
# ==========================================

class DofbotVisualizer:

    # Initializes the visualization state and graphical objects.
    def __init__(self):

        # Command history
        self.history = []
        self.starts = []
        self.total_frames = 0
        self.global_frame = 0
        self.current_frame = 0
        self.active_command = None
        self.command_total = 0

        # Robot state
        self.gripper_angle = GRIPPER_OPEN
        self.controller_running = False
        self.sequence_running = False
        self.paused = False

        # Visualization state
        self.initialized = False
        self.playing = False
        self.show_path = True
        self.updating_slider = False
        self.last_draw_time = 0.0

        # Graphical objects
        self.fig = None
        self.ax = None
        self.info_ax = None
        self.info_text = None

        self.frame_slider = None
        self.play_button = None
        self.first_button = None
        self.prev_button = None
        self.next_button = None
        self.view_button = None
        self.path_button = None
        self.timer = None

        self.link_lines = []
        self.joint_markers = []
        self.joint_labels = []
        self.gripper_lines = []
        self.tool_axes = []

        self.ee_marker = None
        self.ee_label = None
        self.path_line = None
        self.completed_path = None

        self.link_colors = [
            "dimgray",
            "royalblue",
            "darkorange",
            "seagreen",
            "mediumpurple",
            "crimson"
        ]


    # ==========================================
    # COMMAND SEQUENCE
    # ==========================================

    # Clears previous trajectory history and starts a new sequence.
    def begin_sequence(self, total_commands=0):

        if self.timer is not None:
            self.timer.stop()

        self.history = []
        self.starts = []
        self.total_frames = 0
        self.global_frame = 0
        self.current_frame = 0
        self.active_command = None

        self.command_total = total_commands

        self.controller_running = False
        self.sequence_running = True
        self.paused = False
        self.playing = False

    # Marks the command sequence as complete and enables replay.
    def end_sequence(self):

        self.controller_running = False
        self.sequence_running = False
        self.paused = False

        if not self.initialized:
            return

        self.play_button.label.set_text("Replay")

        self.refresh_slider_range()
        self.render(self.global_frame)
        self.process_events()
        
    
    # ==========================================
    # STORE COMMAND TRAJECTORY
    # ==========================================

    # Records a new movement command and its trajectory.
    def new_command(
        self,
        command_type,
        gripper_angle,
        command_number=None,
        command_total=None,
        move_number=None,
        move_total=None
    ):

        if self.timer is not None:
            self.timer.stop()

        q = kinematics.q.copy()
        N = q.shape[1]

        points = np.zeros((N, 7, 3))
        path = np.zeros((N, 3))

        for ii in range(N):

            _, path[ii], points[ii] = (
                kinematics.fwdkin_with_points(q[:, ii])
            )

        command = {
            "type": command_type,
            "number": command_number or len(self.history) + 1,
            "total": command_total or self.command_total,
            "move_number": move_number,
            "move_total": move_total,
            "q": q,
            "path": path,
            "points": points,
            "qstart": q[:, 0].copy(),
            "qend": q[:, -1].copy(),
            "gripper_before": float(gripper_angle),
            "gripper_after": float(gripper_angle),
            "gripper_event": False
        }

        self.starts.append(self.total_frames)
        self.history.append(command)

        self.total_frames += N

        self.active_command = len(self.history) - 1

        self.global_frame = self.starts[self.active_command]
        self.current_frame = 0

        self.gripper_angle = float(gripper_angle)

        self.controller_running = True
        self.paused = False
        self.playing = False

        if not self.initialized:
            self.setup_figure()

        self.play_button.label.set_text("Pause")

        self.refresh_slider_range()
        self.render(self.global_frame)
        self.process_events()
    
    # ==========================================
    # RECORD GRIPPER COMMAND
    # ==========================================

    # Records a gripper command and its associated joint configuration.
    def record_gripper_command(
        self,
        command_type,
        gripper_angle,
        joint_angles,
        command_number=None,
        command_total=None,
        move_total=None
    ):

        # Stop any previous replay
        if self.timer is not None:
            self.timer.stop()

        self.playing = False
        self.paused = False
        self.controller_running = False

        # Current joint configuration
        q = np.asarray(
            joint_angles,
            dtype=float
        ).reshape(5, 1)

        # Forward kinematics
        Rot, Pot, points = (
            kinematics.fwdkin_with_points(q[:, 0])
        )

        # Store stationary robot configuration
        command = {
            "type": command_type,
            "number": command_number or len(self.history) + 1,
            "total": command_total or self.command_total,
            "move_number": None,
            "move_total": move_total,
            "q": q.copy(),
            "path": Pot.reshape(1, 3).copy(),
            "points": points.reshape(1, 7, 3).copy(),
            "qstart": q[:, 0].copy(),
            "qend": q[:, 0].copy(),
            "gripper_before": float(gripper_angle),
            "gripper_after": float(gripper_angle),
            "gripper_event": False
        }

        # ==================================
        # ADD TO COMMAND HISTORY
        # ==================================

        self.starts.append(
            self.total_frames
        )

        self.history.append(
            command
        )

        self.total_frames += 1

        self.active_command = (
            len(self.history) - 1
        )

        self.global_frame = (
            self.starts[self.active_command]
        )

        self.current_frame = 0

        self.gripper_angle = float(
            gripper_angle
        )

        # ==================================
        # UPDATE VISUALIZATION
        # ==================================

        if not self.initialized:
            self.setup_figure()

        self.refresh_slider_range()

        self.render(
            self.global_frame
        )

        self.process_events()

    # Records a gripper state change in the command history.
    def set_gripper_angle(self, angle):

        self.gripper_angle = float(angle)

        if self.active_command is None:
            return

        command = self.history[self.active_command]

        command["gripper_after"] = self.gripper_angle

        if (
            not command["gripper_event"]
            and not np.isclose(
                command["gripper_before"],
                self.gripper_angle
            )
        ):

            command["gripper_event"] = True
            self.total_frames += 1

            self.global_frame = (
                self.starts[self.active_command]
                + command["q"].shape[1]
            )

        if self.initialized:

            self.refresh_slider_range()
            self.render(self.global_frame)
            self.process_events()


    # ==========================================
    # CREATE WINDOW AND OBJECTS
    # ==========================================

    # Creates the Matplotlib window, controls, and graph layout.
    def setup_figure(self):

        self.fig = plt.figure(
            figsize=(14, 8),
            facecolor="#f4f5f7"
        )

        if self.fig.canvas.manager is not None:

            self.fig.canvas.manager.set_window_title(
                "DOFBOT Robot Simulator"
            )

        gs = GridSpec(
            1, 2,
            figure=self.fig,
            width_ratios=[2.1, 1.1],
            left=0.04,
            right=0.98,
            top=0.88,
            bottom=0.24,
            wspace=0.06
        )

        self.ax = self.fig.add_subplot(
            gs[0, 0],
            projection="3d"
        )

        self.info_ax = self.fig.add_subplot(
            gs[0, 1]
        )

        self.info_ax.axis("off")

        self.fig.suptitle(
            "DOFBOT ROBOT SIMULATOR",
            fontsize=17,
            fontweight="bold"
        )

        slider_ax = self.fig.add_axes(
            [0.16, 0.155, 0.70, 0.028]
        )

        self.frame_slider = Slider(
            slider_ax,
            "Sequence frame",
            0,
            max(1, self.total_frames - 1),
            valinit=0,
            valstep=1,
            color="#2878b5"
        )

        button_data = [
            ("Pause", "play_button"),
            ("First Frame", "first_button"),
            ("Prev Cmd", "prev_button"),
            ("Next Cmd", "next_button"),
            ("Reset View", "view_button"),
            ("Hide Path", "path_button")
        ]

        for ii, (label, attribute) in enumerate(button_data):

            button_ax = self.fig.add_axes([
                0.055 + ii * 0.152,
                0.065,
                0.137,
                0.055
            ])

            setattr(
                self,
                attribute,
                Button(
                    button_ax,
                    label,
                    color="#eeeeee"
                )
            )

        self.frame_slider.on_changed(
            self.slider_changed
        )

        self.play_button.on_clicked(
            self.toggle_play
        )

        self.first_button.on_clicked(
            self.first_frame
        )

        self.prev_button.on_clicked(
            self.previous_command
        )

        self.next_button.on_clicked(
            self.next_command
        )

        self.view_button.on_clicked(
            self.reset_view
        )

        self.path_button.on_clicked(
            self.toggle_path
        )

        self.timer = self.fig.canvas.new_timer(
            interval=max(
                20,
                round(
                    1000 * TOTAL_TIME
                    / max(kinematics.N - 1, 1)
                )
            )
        )

        self.timer.add_callback(
            self.advance_frame
        )

        self.setup_artists()

        self.initialized = True

        plt.show(block=False)

    # Initializes the robot's 3D graphical objects.
    def setup_artists(self):

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        self.ax.set_title(
            "Robot Configuration",
            pad=10
        )

        self.reset_axis_limits()

        self.ax.view_init(
            elev=22,
            azim=-58
        )

        self.path_line, = self.ax.plot(
            [], [], [],
            "--",
            color="gray",
            alpha=0.4
        )

        self.completed_path, = self.ax.plot(
            [], [], [],
            color="crimson",
            linewidth=2
        )

        for color in self.link_colors:

            line, = self.ax.plot(
                [], [], [],
                color=color,
                linewidth=7
            )

            self.link_lines.append(line)

        for ii in range(6):

            marker, = self.ax.plot(
                [], [], [],
                "o",
                color="black" if ii == 0 else "#ffb300",
                markersize=9
            )

            label = self.ax.text(
                0, 0, 0,
                "BASE" if ii == 0 else f"J{ii}",
                fontsize=8
            )

            self.joint_markers.append(marker)
            self.joint_labels.append(label)

        self.ee_marker, = self.ax.plot(
            [], [], [],
            "o",
            color="crimson",
            markersize=9
        )

        self.ee_label = self.ax.text(
            0, 0, 0,
            "EE",
            fontsize=9
        )

        for ii in range(2):

            line, = self.ax.plot(
                [], [], [],
                color="black",
                linewidth=5
            )

            self.gripper_lines.append(line)

        for color in ("red", "green", "blue"):

            line, = self.ax.plot(
                [], [], [],
                color=color,
                linewidth=2
            )

            self.tool_axes.append(line)

        self.info_text = self.info_ax.text(
            0.02,
            0.99,
            "",
            transform=self.info_ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.4,
            fontfamily="monospace",
            linespacing=1.06
        )


    # ==========================================
    # GRAPH VIEW
    # ==========================================

    # Restores the default axis limits.
    def reset_axis_limits(self):

        self.ax.set_xlim(-0.31, 0.31)
        self.ax.set_ylim(-0.31, 0.31)
        self.ax.set_zlim(0.0, 0.46)

        self.ax.set_box_aspect(
            (1, 1, 0.75)
        )

    # Restores the default 3D camera view and axis limits.
    def reset_view(self, event):

        self.reset_axis_limits()

        self.ax.view_init(
            elev=22,
            azim=-58
        )

        self.fig.canvas.draw_idle()


    # ==========================================
    # TIMELINE INDEXING
    # ==========================================

    # Updates the timeline slider limits to include recorded frames.
    def refresh_slider_range(self):

        if not self.initialized:
            return

        maximum = max(
            1,
            self.total_frames - 1
        )

        self.frame_slider.valmax = maximum

        self.frame_slider.ax.set_xlim(
            0,
            maximum
        )

    # Identifies the command and local frame for a timeline position.
    def lookup_frame(self, global_frame):

        frame = int(
            np.clip(
                global_frame,
                0,
                self.total_frames - 1
            )
        )

        command_index = (
            bisect.bisect_right(
                self.starts,
                frame
            ) - 1
        )

        command_index = max(
            0,
            command_index
        )

        command = self.history[command_index]

        local_frame = (
            frame - self.starts[command_index]
        )

        gripper_event = (
            local_frame >= command["q"].shape[1]
        )

        local_frame = min(
            local_frame,
            command["q"].shape[1] - 1
        )

        return (
            frame,
            command_index,
            command,
            local_frame,
            gripper_event
        )


    # ==========================================
    # CONTROLLER UPDATE AND RENDER
    # ==========================================

    # Updates the displayed frame during live trajectory execution.
    def update(self, frame):

        if not self.initialized:
            return

        if not plt.fignum_exists(self.fig.number):
            return

        command = self.history[self.active_command]

        frame = int(
            np.clip(
                frame,
                0,
                command["q"].shape[1] - 1
            )
        )

        self.current_frame = frame

        self.global_frame = (
            self.starts[self.active_command]
            + frame
        )

        if frame == command["q"].shape[1] - 1:

            self.controller_running = False

        now = time.monotonic()

        if frame not in (
            0,
            command["q"].shape[1] - 1
        ):

            if now - self.last_draw_time < 1 / MAX_FPS:
                return

        self.render(self.global_frame)

        self.last_draw_time = time.monotonic()

        self.process_events()

    # Renders a recorded robot configuration and gripper state.
    def render(self, global_frame):

        if not self.initialized or not self.history:
            return

        (
            frame,
            command_index,
            command,
            ii,
            gripper_event
        ) = self.lookup_frame(global_frame)

        self.global_frame = frame
        self.current_frame = ii

        self.updating_slider = True

        if int(self.frame_slider.val) != frame:
            self.frame_slider.set_val(frame)

        self.updating_slider = False

        q = command["q"][:, ii]

        Rot, Pot, points = (
            kinematics.fwdkin_with_points(q)
        )

        # Update individual links
        for jj, line in enumerate(self.link_lines):

            segment = points[jj:jj + 2]

            line.set_data(
                segment[:, 0],
                segment[:, 1]
            )

            line.set_3d_properties(
                segment[:, 2]
            )

        # Update joint markers
        for jj, (marker, label) in enumerate(
            zip(
                self.joint_markers,
                self.joint_labels
            )
        ):

            x, y, z = points[jj]

            marker.set_data([x], [y])
            marker.set_3d_properties([z])

            label.set_position((x, y))

            label.set_3d_properties(
                z + 0.012 + (jj % 2) * 0.014
            )

        # Update end effector
        self.ee_marker.set_data(
            [Pot[0]],
            [Pot[1]]
        )

        self.ee_marker.set_3d_properties(
            [Pot[2]]
        )

        self.ee_label.set_position(
            (Pot[0], Pot[1])
        )

        self.ee_label.set_3d_properties(
            Pot[2] + 0.014
        )

        # Update end-effector axes
        for jj, line in enumerate(self.tool_axes):

            segment = np.vstack((
                Pot,
                Pot + 0.035 * Rot[:, jj]
            ))

            line.set_data(
                segment[:, 0],
                segment[:, 1]
            )

            line.set_3d_properties(
                segment[:, 2]
            )

        # Update gripper
        gripper_angle = (
            command["gripper_after"]
            if gripper_event
            else command["gripper_before"]
        )

        grip_fraction = np.clip(
            (
                gripper_angle - GRIPPER_OPEN
            ) / (
                GRIPPER_CLOSED - GRIPPER_OPEN
            ),
            0,
            1
        )

        gap = (
            MAX_GAP
            - grip_fraction * (MAX_GAP - MIN_GAP)
        )

        for jj, side in enumerate((-1, 1)):

            local = np.array([
                [-FINGER_LENGTH, side * gap / 2, 0],
                [0, side * gap / 2, 0]
            ])

            world = (
                Rot @ local.T
            ).T + Pot

            self.gripper_lines[jj].set_data(
                world[:, 0],
                world[:, 1]
            )

            self.gripper_lines[jj].set_3d_properties(
                world[:, 2]
            )

        # Draw full recorded path
        paths = []
        completed = []

        for jj, item in enumerate(self.history):

            path = item["path"]

            paths.append(path)

            if jj < command_index:
                completed.append(path)

            elif jj == command_index:
                completed.append(
                    path[:ii + 1]
                )

        def join_paths(items):

            if not items:
                return np.empty((0, 3))

            result = []

            for path in items:

                if result:
                    result.append(
                        np.full((1, 3), np.nan)
                    )

                result.append(path)

            return np.vstack(result)

        planned = join_paths(paths)
        traveled = join_paths(completed)

        for line, values in (
            (self.path_line, planned),
            (self.completed_path, traveled)
        ):

            line.set_visible(self.show_path)

            line.set_data(
                values[:, 0],
                values[:, 1]
            )

            line.set_3d_properties(
                values[:, 2]
            )

        self.update_info(
            command,
            ii,
            frame,
            q,
            Rot,
            Pot,
            gripper_angle,
            gap,
            gripper_event
        )

        self.fig.canvas.draw_idle()


    # ==========================================
    # PROCESS EVENTS AND USER INPUT
    # ==========================================

    # Updates the robot telemetry and command information panel.
    def update_info(
        self,
        command,
        ii,
        global_frame,
        q,
        Rot,
        Pot,
        gripper_angle,
        gap,
        gripper_event
    ):

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected"
            )

            yaw, pitch, roll = (
                Rotation.from_matrix(Rot).as_euler(
                    "ZYX",
                    degrees=True
                )
            )

        quaternion = (
            Rotation.from_matrix(Rot).as_quat()
        )

        if self.controller_running and self.paused:
            status = "PAUSED"

        elif self.controller_running:
            status = "EXECUTING"

        elif self.playing:
            status = "REPLAYING"

        elif self.sequence_running:
            status = "BETWEEN COMMANDS"

        else:
            status = "REVIEW"

        if gripper_event:

            action = (
                "GRIPPER CLOSE"
                if gripper_angle >= GRIPPER_CLOSED
                else "GRIPPER OPEN"
            )

        elif command["type"] == "MOVE_TO":
            action = "MOVE TO TARGET"

        else:
            action = command["type"]

        qstart = ", ".join(
            f"{value:.1f}"
            for value in command["qstart"]
        )

        qend = ", ".join(
            f"{value:.1f}"
            for value in command["qend"]
        )

        move_number = "-"

        if command["move_number"] is not None:

            move_number = (
                f"{command['move_number']}"
                f" / {command['move_total']}"
            )

        panel = [
            "COMMAND SEQUENCE",
            f"Command       : {command['type']}",
            f"Action        : {action}",
            f"Sequence step : {command['number']} / {command['total']}",
            f"MOVE_TO number: {move_number}",
            f"Status        : {status}",
            f"Timeline frame: {global_frame + 1} / {self.total_frames}",
            f"Command frame : {ii + 1} / {command['q'].shape[1]}",
            "",
            "COMMAND POSITIONS (DEG)",
            f"Original q : [{qstart}]",
            f"Desired q  : [{qend}]",
            "",
            "CURRENT JOINT ANGLES (DEG)",
            *[
                f"Joint {jj + 1}       : {angle:8.2f}"
                for jj, angle in enumerate(q)
            ],
            "",
            "END EFFECTOR POSITION",
            f"X             : {Pot[0]:8.4f} m",
            f"Y             : {Pot[1]:8.4f} m",
            f"Z             : {Pot[2]:8.4f} m",
            "",
            "END EFFECTOR ORIENTATION (ZYX)",
            f"Roll          : {roll:8.2f} deg",
            f"Pitch         : {pitch:8.2f} deg",
            f"Yaw           : {yaw:8.2f} deg",
            "",
            "GRIPPER",
            f"Servo angle   : {gripper_angle:8.1f} deg",
            f"Jaw gap (est.): {gap * 1000:8.1f} mm",
            f"State         : {'CLOSED' if gripper_angle >= GRIPPER_CLOSED else 'OPEN'}",
            "",
            "ROTATION MATRIX",
            *[
                "  " + "  ".join(
                    f"{value: .3f}"
                    for value in row
                )
                for row in Rot
            ],
            "",
            "QUATERNION (X, Y, Z, W)",
            "  " + ", ".join(
                f"{value:.3f}"
                for value in quaternion
            )
        ]

        self.info_text.set_text(
            "\n".join(panel)
        )

    # Processes pending graphical events to keep the window responsive.
    def process_events(self, duration=0.001):

        if not self.initialized:
            return

        if plt.fignum_exists(self.fig.number):

            plt.pause(
                max(duration, 0.001)
            )

    # Holds trajectory execution while the simulation is paused.
    def wait_if_paused(self):

        if not self.paused:
            return 0.0

        if not self.controller_running:
            return 0.0

        pause_start = time.monotonic()

        while (
            self.paused
            and plt.fignum_exists(self.fig.number)
        ):

            self.process_events(0.01)

        return (
            time.monotonic() - pause_start
        )

    # Pauses or resumes live execution or recorded playback.
    def slider_changed(self, value):

        if self.updating_slider:
            return

        if self.sequence_running:
            return

        self.render(
            int(value)
        )

    # Stops recorded playback without changing the robot state.
    def toggle_play(self, event):

        # Live simulation
        if self.sequence_running:

            if self.controller_running:

                self.paused = not self.paused

                self.play_button.label.set_text(
                    "Resume" if self.paused else "Pause"
                )

                self.render(self.global_frame)

            return

        # Recorded history replay
        if not self.history:
            return

        self.playing = not self.playing

        if self.playing:

            if self.global_frame >= self.total_frames - 1:
                self.render(0)

            self.play_button.label.set_text(
                "Pause Replay"
            )

            self.timer.start()

        else:

            self.timer.stop()

            self.play_button.label.set_text(
                "Replay"
            )

        self.render(self.global_frame)

    # Advances to the next frame in the recorded command sequence.
    def advance_frame(self):

        if not self.playing:
            return

        if self.sequence_running:
            return

        if self.global_frame >= self.total_frames - 1:

            self.stop_replay()
            self.render(self.global_frame)

        else:

            self.render(
                self.global_frame + 1
            )

    # Stops the replay of the recorded command sequence.
    def stop_replay(self):

        self.playing = False

        if self.timer is not None:
            self.timer.stop()

        self.play_button.label.set_text(
            "Replay"
        )

    # Jumps to the beginning of the recorded command sequence.
    def first_frame(self, event):

        if self.sequence_running:
            return

        self.stop_replay()

        if self.history:
            self.render(0)

    # Jumps to the beginning of the previous command.
    def previous_command(self, event):

        if self.sequence_running or not self.history:
            return

        self.stop_replay()

        _, index, _, _, _ = (
            self.lookup_frame(self.global_frame)
        )

        if self.global_frame == self.starts[index]:
            index -= 1

        index = max(0, index)

        self.render(
            self.starts[index]
        )

    # Jumps to the beginning of the next command.
    def next_command(self, event):

        if self.sequence_running or not self.history:
            return

        self.stop_replay()

        _, index, _, _, _ = (
            self.lookup_frame(self.global_frame)
        )

        index = min(
            index + 1,
            len(self.history) - 1
        )

        self.render(
            self.starts[index]
        )

    # Shows or hides the end-effector trajectory.
    def toggle_path(self, event):

        self.show_path = not self.show_path

        self.path_button.label.set_text(
            "Hide Path"
            if self.show_path
            else "Show Path"
        )

        self.render(
            self.global_frame
        )


# ==========================================
# RUN SIMULATION
# ==========================================

if __name__ == "__main__":

    import main

    visualizer = DofbotVisualizer()

    try:

        main.run_commands(
            visualizer
        )

        plt.show()

    except KeyboardInterrupt:

        print(
            "\nVisualization stopped by user."
        )

    finally:

        plt.close("all")
