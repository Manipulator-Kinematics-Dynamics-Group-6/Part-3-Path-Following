
import numpy as np
import warnings

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ==========================================
# TRAJECTORY SETTINGS
# ==========================================

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
# JOINT LIMITS (DEGREES)
# ==========================================

# Configure these to match the physical robot.
JOINT_MIN = np.array(
    [0, 0, 0, 0, 0],
    dtype=float
)

JOINT_MAX = np.array(
    [180, 180, 180, 180, 180],
    dtype=float
)


# ==========================================
# INVERSE KINEMATICS SETTINGS
# ==========================================

POSITION_TOL = 1e-4
ORIENTATION_TOL = 1e-3

IK_ATTEMPTS = 24


# ==========================================
# ROTATION MATRICES
# ==========================================

# Generates a 3D rotation matrix about the Z-axis.
def rotz(q):

    q = np.deg2rad(q)

    return np.array([
        [np.cos(q), -np.sin(q), 0],
        [np.sin(q),  np.cos(q), 0],
        [0, 0, 1]
    ])


# Generates a 3D rotation matrix about the Y-axis.
def roty(q):

    q = np.deg2rad(q)

    return np.array([
        [np.cos(q), 0, np.sin(q)],
        [0, 1, 0],
        [-np.sin(q), 0, np.cos(q)]
    ])


# Generates a 3D rotation matrix about the X-axis.
def rotx(q):

    q = np.deg2rad(q)

    return np.array([
        [1, 0, 0],
        [0, np.cos(q), -np.sin(q)],
        [0, np.sin(q),  np.cos(q)]
    ])


# ==========================================
# FORWARD KINEMATICS
# ==========================================

# Computes the end-effector pose and all joint positions.
def fwdkin_with_points(q):

    q = np.asarray(
        q,
        dtype=float
    ).flatten()

    if q.size != 5:
        raise ValueError(
            "q must contain exactly 5 joint angles."
        )

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
    points = np.array([
        p0, p1, p2, p3, p4, p5, Pot
    ])

    return Rot, Pot, points

# Computes the end-effector rotation matrix and position.
def fwdkin_Dofbot(q):

    Rot, Pot, points = fwdkin_with_points(q)

    return Rot, Pot


# ==========================================
# INVERSE KINEMATICS
# ==========================================

# Calculates position and orientation errors for inverse kinematics.
def ik_pose_error(q, Rot_desired, Pot_desired):

    # Forward kinematics of candidate configuration
    Rot_current, Pot_current = fwdkin_Dofbot(q)

    # Position error (meters)
    position_error = (
        Pot_current - Pot_desired
    )

    # Orientation error (rotation vector, radians)
    rotation_error = Rotation.from_matrix(
        Rot_desired @ Rot_current.T
    ).as_rotvec()

    # Normalize position error for optimization.
    # Both error terms are used by the solver.
    return np.concatenate([
        position_error / 0.1,
        rotation_error
    ])

# Solves inverse kinematics numerically using least squares.
# Returns valid joint configurations within the specified limits.
def invkin_Dofbot(Rot, Pot, qstart=None):

    """
    Calculate joint configurations that reproduce
    the desired end-effector pose.

    Parameters
    ----------
    Rot : ndarray, shape (3, 3)
        Desired rotation matrix.

    Pot : ndarray, shape (3,)
        Desired position in meters.

    qstart : ndarray, shape (5,), optional
        Current joint configuration, in degrees.

    Returns
    -------
    q : ndarray, shape (5, n)
        Valid IK solutions in degrees.

        Returns an empty (5, 0) array if no
        verified solution is found.
    """

    Rot = np.asarray(
        Rot,
        dtype=float
    ).reshape(3, 3)

    Pot = np.asarray(
        Pot,
        dtype=float
    ).reshape(3)

    # ==================================
    # VALIDATE DESIRED POSE
    # ==================================

    if not (
        np.all(np.isfinite(Rot))
        and np.all(np.isfinite(Pot))
    ):
        raise ValueError(
            "Desired pose must contain finite values."
        )

    if not np.allclose(
        Rot.T @ Rot,
        np.eye(3),
        atol=1e-5
    ):
        raise ValueError(
            "Rot must be an orthogonal rotation matrix."
        )

    if not np.isclose(
        np.linalg.det(Rot),
        1.0,
        atol=1e-5
    ):
        raise ValueError(
            "Rot must have determinant +1."
        )

    # ==================================
    # INITIAL JOINT CONFIGURATIONS
    # ==================================

    qcenter = (
        JOINT_MIN + JOINT_MAX
    ) / 2

    seeds = [
        qcenter.copy()
    ]

    if qstart is not None:

        qstart = np.asarray(
            qstart,
            dtype=float
        ).flatten()

        if qstart.size != 5:
            raise ValueError(
                "qstart must contain 5 joint angles."
            )

        if not np.all(np.isfinite(qstart)):
            raise ValueError(
                "qstart must contain finite angles."
            )

        seeds.insert(
            0,
            np.clip(qstart, JOINT_MIN, JOINT_MAX)
        )

    # Additional initial guesses to find different inverse-kinematic solutions
    rng = np.random.default_rng(0)

    for ii in range(IK_ATTEMPTS):

        seeds.append(
            rng.uniform(
                JOINT_MIN,
                JOINT_MAX
            )
        )

    solutions = []

    # ==================================
    # SOLVE INVERSE KINEMATICS
    # ==================================

    for seed in seeds:

        result = least_squares(
            ik_pose_error,
            seed,
            args=(Rot, Pot),
            bounds=(JOINT_MIN, JOINT_MAX),
            max_nfev=250,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10
        )

        q_candidate = result.x

        # ==================================
        # VERIFY WITH FORWARD KINEMATICS
        # ==================================

        Rot_check, Pot_check = fwdkin_Dofbot(
            q_candidate
        )

        position_error = np.linalg.norm(
            Pot_check - Pot
        )

        orientation_error = Rotation.from_matrix(
            Rot @ Rot_check.T
        ).magnitude()

        if (
            position_error <= POSITION_TOL
            and orientation_error <= ORIENTATION_TOL
        ):

            # Avoid duplicate solutions
            duplicate = any(
                np.linalg.norm(
                    existing - q_candidate
                ) < 0.1
                for existing in solutions
            )

            if not duplicate:

                solutions.append(
                    q_candidate.copy()
                )

    # ==================================
    # RETURN VALID SOLUTIONS
    # ==================================

    if len(solutions) == 0:

        return np.empty(
            (5, 0),
            dtype=float
        )

    # Sort solutions by distance from qstart
    if qstart is not None:

        solutions.sort(
            key=lambda q:
                np.linalg.norm(q - qstart)
        )

    return np.column_stack(solutions)


# ==========================================
# TRAJECTORY GENERATION
# ==========================================

q = np.zeros((5, N))
Rot = np.zeros((3, 3, N))
eulerot = np.zeros((3, N))
Pot = np.zeros((3, N))

# Generates a linear joint-space trajectory between two configurations.
def generate_trajectory(qstart, qend):

    qstart = np.asarray(
        qstart,
        dtype=float
    ).flatten()

    qend = np.asarray(
        qend,
        dtype=float
    ).flatten()

    if qstart.size != 5 or qend.size != 5:
        raise ValueError(
            "qstart and qend must contain 5 joint angles."
        )

    for ii in range(N):

        # Joint-space linear interpolation
        q[:, ii] = (
            qstart * (1 - lam[ii])
            + qend * lam[ii]
        )

        # Forward kinematics
        Rot[:, :, ii], Pot[:, ii] = (
            fwdkin_Dofbot(q[:, ii])
        )

        # Euler angles
        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected"
            )

            euler_angles = Rotation.from_matrix(
                Rot[:, :, ii]
            ).as_euler(
                "ZYX",
                degrees=True
            )

        eulerot[:, ii] = (
            (euler_angles + 180) % 360 - 180
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # ==================================
    # TEST FORWARD KINEMATICS
    # ==================================

    qstart = np.array(
        [90, 90, 90, 90, 90],
        dtype=float
    )

    qend = np.array(
        [45, 120, 140, 100, 95],
        dtype=float
    )

    Rot_desired, Pot_desired = (
        fwdkin_Dofbot(qend)
    )

    print("Desired end-effector position:")
    print(Pot_desired)

    print("Desired end-effector rotation:")
    print(Rot_desired)

    # ==================================
    # TEST INVERSE KINEMATICS
    # ==================================

    q_solutions = invkin_Dofbot(
        Rot_desired,
        Pot_desired,
        qstart
    )

    print("\nInverse kinematics solutions:")

    if q_solutions.shape[1] == 0:

        print("No verified solutions found.")

    else:

        print(
            np.round(q_solutions, 3)
        )

        q_solution = q_solutions[:, 0]

        Rot_check, Pot_check = (
            fwdkin_Dofbot(q_solution)
        )

        position_error = np.linalg.norm(
            Pot_check - Pot_desired
        )

        orientation_error = Rotation.from_matrix(
            Rot_desired @ Rot_check.T
        ).magnitude()

        print(
            f"Position error: {position_error * 1000:.6f} mm"
        )

        print(
            f"Orientation error: "
            f"{np.rad2deg(orientation_error):.6f} deg"
        )

    # ==================================
    # GENERATE JOINT TRAJECTORY
    # ==================================

    generate_trajectory(
        qstart,
        qend
    )

    # ==================================
    # PLOT JOINT POSITIONS
    # ==================================

    plt.figure()

    plt.plot(
        lam,
        q.T,
        linewidth=2
    )

    plt.xlabel("Lambda")
    plt.ylabel("Joint angles (degrees)")
    plt.title("Joint Positions")

    plt.legend([
        "Joint 1",
        "Joint 2",
        "Joint 3",
        "Joint 4",
        "Joint 5"
    ])

    plt.grid()

    # ==================================
    # PLOT END EFFECTOR ORIENTATION
    # ==================================

    plt.figure()

    plt.plot(
        lam,
        eulerot.T,
        linewidth=2
    )

    plt.xlabel("Lambda")
    plt.ylabel("Rotation (degrees)")
    plt.title("End-Effector Orientation")

    plt.legend([
        "Yaw",
        "Pitch",
        "Roll"
    ])

    plt.grid()

    # ==================================
    # PLOT END EFFECTOR POSITION
    # ==================================

    plt.figure()

    plt.plot(
        lam,
        Pot.T,
        linewidth=2
    )

    plt.xlabel("Lambda")
    plt.ylabel("Position (meters)")
    plt.title("End-Effector Position")

    plt.legend([
        "X",
        "Y",
        "Z"
    ])

    plt.grid()

    plt.show()
