
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button

import dofbot_kinematics as robot


# ==========================================
# GRIPPER SETTINGS
# ==========================================

GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130

FINGER_LENGTH = 0.045

MAX_GAP = 0.045
MIN_GAP = 0.008


# ==========================================
# DOFBOT VISUALIZER
# ==========================================

class DofbotVisualizer:

    def __init__(self):

        self.q = None
        self.Pot = None
        self.N = 0

        self.gripper_angle = GRIPPER_OPEN

        self.current_command = "Ready"

        self.all_points = None
        self.ee_path = None
        self.axis_limits = None

        self.current_frame = 0
        self.playing = False
        self.initialized = False

        self.fig = None
        self.ax = None

        self.frame_slider = None
        self.play_button = None
        self.reset_button = None

        self.timer = None

        self.updating_slider = False

    # ======================================
    # LOAD TRAJECTORY
    # ======================================

    def load_trajectory(self):

        self.q = robot.q.copy()
        self.Pot = robot.Pot.copy()
        self.N = robot.N

        self.all_points = np.zeros((self.N, 7, 3))

        for i in range(self.N):

            _, _, points = robot.fwdkin_with_points(
                self.q[:, i]
            )

            self.all_points[i] = points

        self.ee_path = self.Pot.T.copy()

        minimum = np.min(
            self.all_points.reshape(-1, 3),
            axis=0
        )

        maximum = np.max(
            self.all_points.reshape(-1, 3),
            axis=0
        )

        center = (minimum + maximum) / 2

        limit = max(
            np.max(maximum - minimum) * 0.7,
            0.12
        )

        self.axis_limits = (
            center[0] - limit,
            center[0] + limit,
            center[1] - limit,
            center[1] + limit,
            center[2] - limit,
            center[2] + limit
        )

    # ======================================
    # INITIALIZE FIGURE
    # ======================================

    def setup_figure(self):

        self.fig = plt.figure(figsize=(12, 10))

        self.fig.canvas.manager.set_window_title(
            "Dofbot Simulator"
        )

        self.ax = self.fig.add_axes(
            [0.08, 0.28, 0.84, 0.67],
            projection="3d"
        )

        slider_ax = self.fig.add_axes(
            [0.20, 0.17, 0.60, 0.035]
        )

        self.frame_slider = Slider(
            slider_ax,
            "Frame",
            0,
            self.N - 1,
            valinit=0,
            valstep=1
        )

        play_ax = self.fig.add_axes(
            [0.25, 0.07, 0.15, 0.055]
        )

        reset_ax = self.fig.add_axes(
            [0.60, 0.07, 0.15, 0.055]
        )

        self.play_button = Button(play_ax, "Play")
        self.reset_button = Button(reset_ax, "Reset")

        self.timer = self.fig.canvas.new_timer(
            interval=50
        )

        self.timer.add_callback(self.advance_frame)

        self.frame_slider.on_changed(
            self.slider_changed
        )

        self.play_button.on_clicked(
            self.toggle_play
        )

        self.reset_button.on_clicked(
            self.reset_frame
        )

        self.initialized = True

    # ======================================
    # START
    # ======================================

    def start(self):

        if not self.initialized:

            self.load_trajectory()
            self.setup_figure()

        self.draw_robot(0)

        plt.show(block=False)
        plt.pause(0.1)

    # ======================================
    # NEW COMMAND
    # ======================================

    def new_command(self, command_type, gripper_angle):

        self.playing = False

        if self.initialized:

            self.timer.stop()

            self.play_button.label.set_text("Play")

        self.load_trajectory()

        self.current_command = command_type
        self.gripper_angle = gripper_angle

        self.current_frame = 0

        if not self.initialized:

            self.setup_figure()

        else:

            self.frame_slider.valmin = 0
            self.frame_slider.valmax = self.N - 1

            self.frame_slider.ax.set_xlim(
                0, self.N - 1
            )

            self.frame_slider.set_val(0)

        self.draw_robot(0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    # ======================================
    # GRIPPER UPDATE
    # ======================================

    def set_gripper_angle(self, angle):

        self.gripper_angle = angle

        if self.initialized:

            self.draw_robot(self.current_frame)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            plt.pause(0.001)

    # ======================================
    # DRAW ROBOT
    # ======================================

    def draw_robot(self, frame):

        if self.q is None:
            return

        frame = int(np.clip(frame, 0, self.N - 1))
        self.current_frame = frame

        self.ax.clear()

        joint_angles = self.q[:, frame]

        Rot, Pot, points = robot.fwdkin_with_points(
            joint_angles
        )

        # ==================================
        # GRIPPER GAP
        # ==================================

        grip_fraction = (
            self.gripper_angle - GRIPPER_OPEN
        ) / (
            GRIPPER_CLOSED - GRIPPER_OPEN
        )

        grip_fraction = np.clip(grip_fraction, 0, 1)

        gap = MAX_GAP - grip_fraction * (
            MAX_GAP - MIN_GAP
        )

        # ==================================
        # EE TRAJECTORY
        # ==================================

        self.ax.plot(
            self.ee_path[:, 0],
            self.ee_path[:, 1],
            self.ee_path[:, 2],
            "--",
            color="gray",
            alpha=0.3,
            linewidth=1.5
        )

        # ==================================
        # SEPARATE LINKS
        # ==================================

        link_colors = [
            "dimgray",
            "royalblue",
            "darkorange",
            "seagreen",
            "purple",
            "crimson"
        ]

        for i in range(len(points) - 1):

            p1 = points[i]
            p2 = points[i + 1]

            self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=link_colors[i],
                linewidth=8,
                solid_capstyle="round"
            )

        # ==================================
        # JOINT MARKERS AND LABELS
        # ==================================

        for i in range(len(points)):

            p = points[i]

            if i == 0:

                color = "black"
                name = "Base"
                size = 150

            elif i == 6:

                color = "red"
                name = "EE"
                size = 150

            else:

                color = "orange"
                name = f"J{i}"
                size = 120

            self.ax.scatter(
                p[0], p[1], p[2],
                color=color,
                edgecolors="black",
                linewidths=1.2,
                s=size
            )

            self.ax.text(
                p[0],
                p[1],
                p[2] + 0.012,
                name,
                fontsize=9,
                fontweight="bold"
            )

        # ==================================
        # JOINT ANGLES
        # ==================================

        for i in range(5):

            p = points[i + 1]

            self.ax.text(
                p[0],
                p[1],
                p[2] - 0.018,
                f"J{i+1}: {joint_angles[i]:.1f} deg",
                fontsize=8,
                color="darkred",
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="none"
                )
            )

        # ==================================
        # GRIPPER
        # ==================================

        for side in [-1, 1]:

            local_points = np.array([
                [0, side * gap / 2, 0],
                [-FINGER_LENGTH, side * gap / 2, 0]
            ])

            world_points = (
                Rot @ local_points.T
            ).T + Pot

            self.ax.plot(
                world_points[:, 0],
                world_points[:, 1],
                world_points[:, 2],
                color="black",
                linewidth=5
            )

        # ==================================
        # BASE AND EE
        # ==================================

        self.ax.scatter(
            0, 0, 0,
            color="black",
            s=150
        )

        self.ax.scatter(
            Pot[0], Pot[1], Pot[2],
            color="red",
            s=100
        )

        # ==================================
        # AXES
        # ==================================

        xmin, xmax, ymin, ymax, zmin, zmax = (
            self.axis_limits
        )

        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(zmin, zmax)

        self.ax.set_box_aspect([1, 1, 1])

        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_zlabel("Z (meters)")

        self.ax.set_title(
            f"Dofbot | {self.current_command}\n"
            f"Frame {frame+1}/{self.N}\n"
            f"EE Position: {np.round(Pot, 4)} m\n"
            f"Gripper: {self.gripper_angle} degrees"
        )

        self.ax.view_init(elev=25, azim=-55)

        self.fig.canvas.draw_idle()

    # ======================================
    # UPDATE FROM CONTROLLER
    # ======================================

    def update(self, frame):

        if not self.initialized:
            return

        self.draw_robot(frame)

        self.updating_slider = True

        self.frame_slider.set_val(frame)

        self.updating_slider = False

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    # ======================================
    # SLIDER
    # ======================================

    def slider_changed(self, value):

        if self.updating_slider:
            return

        if self.playing:

            self.playing = False
            self.timer.stop()

            self.play_button.label.set_text("Play")

        self.draw_robot(int(value))

    # ======================================
    # PLAYBACK
    # ======================================

    def advance_frame(self):

        if self.N == 0:
            return

        current = int(self.frame_slider.val)

        if current >= self.N - 1:
            self.frame_slider.set_val(0)
        else:
            self.frame_slider.set_val(current + 1)

    def toggle_play(self, event):

        self.playing = not self.playing

        if self.playing:

            self.play_button.label.set_text("Pause")
            self.timer.start()

        else:

            self.play_button.label.set_text("Play")
            self.timer.stop()

    # ======================================
    # RESET
    # ======================================

    def reset_frame(self, event):

        self.playing = False
        self.timer.stop()

        self.play_button.label.set_text("Play")

        self.frame_slider.set_val(0)


# ==========================================
# RUN SIMULATION
# ==========================================

if __name__ == "__main__":

    import main

    visualizer = DofbotVisualizer()

    visualizer.start()

    main.run_commands(visualizer)

    plt.show()
