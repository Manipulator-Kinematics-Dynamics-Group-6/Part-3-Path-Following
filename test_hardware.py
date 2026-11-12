import time
from Arm_Lib import Arm_Device
import import_ipynb # only needed if running with IPYNB files on Jupyter, comment out otherwise 
import dofbot_kinematics as kinematics
import dofbot_controller as controller

arm = Arm_Device()

if __name__ == "__main__":
    print("Servo 1:", arm.Arm_serial_servo_read(1))
    print("Servo 2:", arm.Arm_serial_servo_read(2))
    print("Servo 3:", arm.Arm_serial_servo_read(3))
    print("Servo 4:", arm.Arm_serial_servo_read(4))
    print("Servo 5:", arm.Arm_serial_servo_read(5))
    print("Gripper:", arm.Arm_serial_servo_read(6))
    
    # Initialize joint angles and use forward kinematics to calculate the end-effector position and orientation
    q = [45, 120, 140, 100, 95] 
#     q = [90, 90, 90, 90, 90]
    fwd_kinematics = kinematics.fwdkin_Dofbot(q)
    rot = fwd_kinematics[0]
    pot = fwd_kinematics[1]
    
    # Print the calculated end-effector position and orientation
    print("End-effector position (P_0T):", pot)
    print("End-effector orientation (R_0T):\n", rot)

    # Command the robot arm to move to the calculated position
    arm.Arm_serial_servo_write6(q[0], q[1], q[2], q[3], q[4], 90, 1000)
    time.sleep(.01)
    
    # Print the actual angles of all servos after the movement
    print("Actual angles of all servos after movement:")
    for i in range(1,7):
        print("Servo", i, ":", arm.Arm_serial_servo_read(i))
        time.sleep(.01)
    
    
