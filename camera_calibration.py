
import cv2
import numpy as np
from pathlib import Path

# ----------------------------------------
# Camera calibration settings
# ----------------------------------------
CAMERA_ID = 0                 # Default laptop webcam
CHECKERBOARD = (8, 6)         # Internal corners, not squares
NUM_IMAGES = 15               # Number of images to capture
SQUARE_SIZE = 1.0             # Relative units

output = Path("calibration_images")
output.mkdir(exist_ok=True)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ----------------------------------------
# Generate reference 3D checkerboard points
# ----------------------------------------
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2) * SQUARE_SIZE

object_points = []
image_points = []

# ----------------------------------------
# Open camera
# ----------------------------------------
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

count = 0
image_size = None

print("Press SPACE to capture a valid checkerboard.")
print("Press Q to finish.")

while count < NUM_IMAGES:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, None
    )

    display = frame.copy()

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        cv2.drawChessboardCorners(
            display, CHECKERBOARD, corners, found
        )

    status = f"Images: {count}/{NUM_IMAGES}"
    status += " | SPACE: capture | Q: finish"

    cv2.putText(
        display, status, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 255, 0), 2
    )

    cv2.imshow("Camera Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:
        if found:
            count += 1
            object_points.append(objp.copy())
            image_points.append(corners.copy())
            image_size = gray.shape[::-1]

            cv2.imwrite(
                str(output / f"image_{count:02d}.png"),
                frame
            )
            cv2.imwrite(
                str(output / f"corners_{count:02d}.png"),
                display
            )

            print(f"Captured image {count}")
        else:
            print("Checkerboard not detected. Try again.")

cap.release()
cv2.destroyAllWindows()

# ----------------------------------------
# Calculate camera calibration
# ----------------------------------------
if len(object_points) < 3:
    raise RuntimeError(
        "Not enough valid images. Capture more views."
    )

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)

print("\nIntrinsic camera matrix K:")
print(K)

print("\nDistortion coefficients:")
print(dist)

print("\nRMS reprojection error (pixels):")
print(rms)

np.savetxt("intrinsic_matrix.txt", K, fmt="%.6f")

print("\nSaved intrinsic_matrix.txt")
print("Saved calibration images in calibration_images/")
