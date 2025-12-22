from gpiozero import Servo
from time import sleep

servo = Servo(18)  # GPIO 18

# Move instantly to 0° (min)
servo.min()
print("0°")
sleep(5)

# Move instantly to 180° (max)
#servo.max()
print("180°")

servo.max()

sleep(5)
servo.min()

servo.detach()  # release servo
