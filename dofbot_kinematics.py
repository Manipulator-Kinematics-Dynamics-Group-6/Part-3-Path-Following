import numpy as np
from scipy.spatial.transform import Rotation
import warnings

# ==========================================
# INITIAL AND FINAL CONFIGURATIONS
# ==========================================

# qstart = np.array([90, 90, 90, 90, 90], dtype=float)
# qend = np.array([90, 30, 45, 135, 110], dtype=float)

N = 100
lam = np.linspace(0, 1, N)

# ==========================================
# LINK LENGTHS (METERS)
# ==========================================

l0 = 0.061
l1 = 0.0435
l2 = 0.08285
l3 = 0.08285
l4 = 0.07385
l5 = 0.05457

LINK_LENGTHS = [l0, l1, l2, l3, l4, l5]

# ==========================================
# ROTATION MATRICES
# ==========================================

def rotz(q):
    q = np.deg2rad(q)
    return np.array([
        [np.cos(q), -np.sin(q), 0],
        [np.sin(q),  np.cos(q), 0],
        [0, 0, 1]
    ])


def roty(q):
    q = np.deg2rad(q)
    return np.array([
        [np.cos(q), 0, np.sin(q)],
        [0, 1, 0],
        [-np.sin(q), 0, np.cos(q)]
    ])


def rotx(q):
    q = np.deg2rad(q)
    return np.array([
        [1, 0, 0],
        [0, np.cos(q), -np.sin(q)],
        [0, np.sin(q), np.cos(q)]
    ])


# ==========================================
# FORWARD KINEMATICS WITH JOINT POSITIONS
# ==========================================

def fwdkin_with_points(q):

    q = np.asarray(q, dtype=float).flatten()

    if q.size != 5:
        raise ValueError("q must contain exactly 5 joint angles.")

    # Rotation matrices
    R01 = rotz(q[0])
    R12 = roty(-q[1])
    R23 = roty(-q[2])
    R34 = roty(-q[3])
    R45 = rotx(-q[4])

    # Position vectors
    P01 = (l0 + l1) * np.array([0, 0, 1])
    P12 = np.zeros(3)
    P23 = l2 * np.array([1, 0, 0])
    P34 = -l3 * np.array([0, 0, 1])
    P45 = np.zeros(3)
    P5T = -(l4 + l5) * np.array([1, 0, 0])

    # Cumulative rotations
    R02 = R01 @ R12
    R03 = R02 @ R23
    R04 = R03 @ R34
    R05 = R04 @ R45

    # Joint positions
    p0 = np.zeros(3)
    p1 = p0 + P01
    p2 = p1 + R01 @ P12
    p3 = p2 + R02 @ P23
    p4 = p3 + R03 @ P34
    p5 = p4 + R04 @ P45

    # End-effector position
    Pot = p5 + R05 @ P5T

    # End-effector orientation
    Rot = R05

    # All positions, from base to end effector
    points = np.array([p0, p1, p2, p3, p4, p5, Pot])

    return Rot, Pot, points


# ==========================================
# ORIGINAL FORWARD KINEMATICS INTERFACE
# ==========================================

def fwdkin_Dofbot(q):

    Rot, Pot, points = fwdkin_with_points(q)

    return Rot, Pot


# ==========================================
# TRAJECTORY GENERATION
# ==========================================

q = np.zeros((5, N))
Rot = np.zeros((3, 3, N))
eulerot = np.zeros((3, N))
Pot = np.zeros((3, N))

def generate_trajectory(qstart, qend):

    for ii in range(N):

        # Joint interpolation
        q[:, ii] = (
            qstart * (1 - lam[ii])
            + qend * lam[ii]
        )

        # Forward kinematics
        Rot[:, :, ii], Pot[:, ii] = fwdkin_Dofbot(q[:, ii])

        # Euler angles
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected"
            )

            euler_angles = Rotation.from_matrix(
                Rot[:, :, ii]
            ).as_euler('ZYX', degrees=True)

        eulerot[:, ii] = (euler_angles + 180) % 360 - 180


# ==========================================
# OPTIONAL: PLOT ORIGINAL GRAPHS
# ==========================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Joint positions
    plt.figure()
    plt.plot(lam, q.T, linewidth=2)
    plt.xlabel("Lambda")
    plt.ylabel("Joint angles (degrees)")
    plt.title("Joint Positions")
    plt.legend(["Joint 1", "Joint 2", "Joint 3",
                "Joint 4", "Joint 5"])
    plt.grid()

    # Euler orientation
    plt.figure()
    plt.plot(lam, eulerot.T, linewidth=2)
    plt.xlabel("Lambda")
    plt.ylabel("Rotation (degrees)")
    plt.title("EE Orientation")
    plt.legend(["Z", "Y", "X"])
    plt.grid()

    # End-effector position
    plt.figure()
    plt.plot(lam, Pot.T, linewidth=2)
    plt.xlabel("Lambda")
    plt.ylabel("Position (meters)")
    plt.title("EE Position")
    plt.legend(["X", "Y", "Z"])
    plt.grid()

    plt.show()
