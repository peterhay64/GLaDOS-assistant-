from gpiozero import Servo
import time 

servo = Servo(18)

servo.mid()
time.sleep(0.3)
servo.detach()

while True:
    input = input("a to min, d to max ")

    if input == "a":
        servo.min()
        time.sleep(0.3)
        servo.detach()
    elif input == "d":
        servo.max()
        time.sleep(0.3)
        servo.detach()
    elif input == "q":
        break

    
