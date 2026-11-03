from Arm_Lib import Arm_Device

arm = Arm_Device()

if __name__ == "__main__":
    print("Servo 1:", arm.Arm_serial_servo_read(1))
    print("Servo 2:", arm.Arm_serial_servo_read(2))
    print("Servo 3:", arm.Arm_serial_servo_read(3))
    print("Servo 4:", arm.Arm_serial_servo_read(4))
    print("Servo 5:", arm.Arm_serial_servo_read(5))
    print("Gripper:", arm.Arm_serial_servo_read(6))