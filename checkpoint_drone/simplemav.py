# takeoff_mavsdk.py
import asyncio
from mavsdk import System
from gpiozero import Servo
from time import sleep

servo = Servo(18)  # GPIO 18
sleep(1)
servo.min()


async def run():
    drone = System()
    # If your Pixhawk appears as /dev/ttyACM0 over USB:
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")

    print("Waiting for vehicle...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    print("Waiting for heartbeat / ready...")
    # (optional) wait for health checks to be OK in telemetry before arming

    print("Arming...")
    await drone.action.arm()

    # send takeoff (this will put PX4 into takeoff/position control)
    print("Takeoff...")
    await drone.action.takeoff()
    sleep(15)
    
    print("dropping")
    # Move instantly to 180° (max)
    servo.max()
    print("180°")
    
    
    # stay up for 10 s then land (example)
    #await asyncio.sleep(10)
    print("Landing...")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
