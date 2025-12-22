from gpiozero import Servo
from time import sleep

servo = Servo(18)  # GPIO 18
sleep(5)
servo.max()
sleep(5)
servo.min()
