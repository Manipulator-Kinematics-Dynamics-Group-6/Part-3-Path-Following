
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
# PRE-CALCULATE ROBOT POSITIONS
# ==========================================

all_points = np.zeros((robot.N, 7, 3))

for i in range(robot.N):

    _, _, points = robot.fwdkin_with_points(
        robot.q[:, i]
    )

    all_points[i] = points


ee_path = robot.Pot.T

# ==========================================
# FIGURE SETUP
# ==========================================

fig = plt.figure(figsize=(11, 9))

# Leave room at bottom for slider and buttons
ax = fig.add_axes([0.08, 0.28, 0.84, 0.67],
                  projection='3d')

# Slider axis
slider_ax = fig.add_axes([0.20, 0.17, 0.60, 0.035])

frame_slider = Slider(
    slider_ax,
    'Frame',
    0,
    robot.N - 1,
    valinit=0,
    valstep=1
)

# Button axes
play_ax = fig.add_axes([0.25, 0.07, 0.15, 0.055])
reset_ax = fig.add_axes([0.60, 0.07, 0.15, 0.055])

play_button = Button(play_ax, 'Play')
reset_button = Button(reset_ax, 'Reset')


# ==========================================
# ROBOT DRAWING FUNCTION
# ==========================================

def draw_robot(frame):

    frame = int(frame)

    ax.clear()

    # Current joint angles
    joint_angles = robot.q[:, frame]

    # Current end-effector pose
    Rot, Pot, points = robot.fwdkin_with_points(
        joint_angles
    )

    # Gripper position
    grip_angle = (
        GRIPPER_OPEN * (1 - robot.lam[frame])
        + GRIPPER_CLOSED * robot.lam[frame]
    )

    grip_fraction = (
        (grip_angle - GRIPPER_OPEN)
        / (GRIPPER_CLOSED - GRIPPER_OPEN)
    )

    gap = MAX_GAP - grip_fraction * (
        MAX_GAP - MIN_GAP
    )

    # ======================================
    # END EFFECTOR TRAJECTORY
    # ======================================

    ax.plot(
        ee_path[:, 0],
        ee_path[:, 1],
        ee_path[:, 2],
        '--',
        color='gray',
        alpha=0.4,
        linewidth=1.5
    )

    # Current trajectory position
    ax.scatter(
        Pot[0], Pot[1], Pot[2],
        color='red',
        s=100
    )

    # ======================================
    # ROBOT LINKS
    # ======================================

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        '-o',
        color='royalblue',
        linewidth=7,
        markersize=9,
        markerfacecolor='orange',
        markeredgecolor='black'
    )

    # ======================================
    # GRIPPER
    # ======================================

    for side in [-1, 1]:

        local_points = np.array([
            [0, side * gap / 2, 0],
            [-FINGER_LENGTH, side * gap / 2, 0]
        ])

        world_points = (
            Rot @ local_points.T
        ).T + Pot

        ax.plot(
            world_points[:, 0],
            world_points[:, 1],
            world_points[:, 2],
            color='black',
            linewidth=5
        )

    # Base
    ax.scatter(0, 0, 0, color='black', s=120)

    # ======================================
    # AXIS SETTINGS
    # ======================================

    all_xyz = all_points.reshape(-1, 3)

    minimum = np.min(all_xyz, axis=0)
    maximum = np.max(all_xyz, axis=0)

    center = (minimum + maximum) / 2

    limit = max(np.max(maximum - minimum) * 0.7, 0.12)

    ax.set_xlim(center[0] - limit, center[0] + limit)
    ax.set_ylim(center[1] - limit, center[1] + limit)
    ax.set_zlim(center[2] - limit, center[2] + limit)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')

    ax.set_title(
        f'Dofbot | Frame {frame+1}/{robot.N}\n'
        f'Joint angles: {np.round(joint_angles, 1)} degrees\n'
        f'EE Position: {np.round(Pot, 4)} m'
    )

    ax.view_init(elev=25, azim=-55)

    fig.canvas.draw_idle()


# ==========================================
# SLIDER CALLBACK
# ==========================================

def slider_changed(value):

    draw_robot(int(value))


frame_slider.on_changed(slider_changed)


# ==========================================
# PLAY / PAUSE
# ==========================================

playing = False

timer = fig.canvas.new_timer(interval=50)


def advance_frame():

    current = int(frame_slider.val)

    if current >= robot.N - 1:
        frame_slider.set_val(0)
    else:
        frame_slider.set_val(current + 1)


def toggle_play(event):

    global playing

    playing = not playing

    if playing:
        play_button.label.set_text('Pause')
        timer.start()
    else:
        play_button.label.set_text('Play')
        timer.stop()


timer.add_callback(advance_frame)

play_button.on_clicked(toggle_play)


# ==========================================
# RESET BUTTON
# ==========================================

def reset_frame(event):

    global playing

    playing = False

    timer.stop()

    play_button.label.set_text('Play')

    frame_slider.set_val(0)


reset_button.on_clicked(reset_frame)


# ==========================================
# INITIAL DISPLAY
# ==========================================

draw_robot(0)

plt.show()
