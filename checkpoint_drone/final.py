#!/usr/bin/env python3
"""
takeoff_move_drop_return.py

Sequence:
 - connect to vehicle
 - arm and takeoff to TAKEOFF_ALTITUDE
 - switch to offboard, move forward at FORWARD_SPEED for FORWARD_DURATION
 - stop, actuate servo to drop payload
 - move backward at FORWARD_SPEED for FORWARD_DURATION (to return approx. to start)
 - stop offboard and land
"""

import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from gpiozero import Servo
import sys
import time

# CONFIG — tune these for your vehicle / mission
SERVO_GPIO_PIN = 18            # GPIO pin for servo (as your original)
TAKEOFF_ALTITUDE = 5.0         # meters
FORWARD_SPEED = 1.0            # meters/second (positive => forward)
FORWARD_DURATION = 5.0         # seconds to move forward (tune for distance)
HOVER_AFTER_TAKEOFF = 7.0      # seconds to stabilize after takeoff
HOVER_AFTER_MOVE = 1.0         # seconds to stabilize after motion
SERVO_DROP_TIME = 1.0          # seconds for servo to move to 'drop' position

async def run():
    # initialize servo (synchronous)
    servo = Servo(SERVO_GPIO_PIN)
    # ensure servo starts in closed (or safe) position
    try:
        servo.max()
    except Exception as e:
        print(f"Warning: servo init failed: {e}")

    drone = System()

    # Adjust connection string if your vehicle uses UDP/serial different from example
    # Example: serial:///dev/ttyACM0:115200 (your original)
    # Example for SITL: udp://:14540
    system_address = "serial:///dev/ttyACM0:115200"
    print(f"Connecting to drone at {system_address} ...")
    await drone.connect(system_address=system_address)

    # wait for connection
    print("Waiting for vehicle connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to vehicle.")
            break

    # optional: set desired takeoff altitude
    try:
        print(f"Setting takeoff altitude to {TAKEOFF_ALTITUDE} m")
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
    except Exception as e:
        print(f"Could not set takeoff altitude (continuing): {e}")

    # ARM
    print("Arming...")
    await drone.action.arm()

    # TAKEOFF
    print("Taking off...")
    await drone.action.takeoff()
    time.sleep(5)

    # wait for vehicle to climb / stabilize
    print(f"Waiting {HOVER_AFTER_TAKEOFF} s to reach altitude and stabilize...")
    await asyncio.sleep(HOVER_AFTER_TAKEOFF)

    # ------ Switch to OFFBOARD mode to command velocities -------
    # Must set an initial setpoint before starting offboard
    try:
        print("Setting initial offboard setpoint (zero velocity).")
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        print("Starting offboard mode...")
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Failed to start offboard mode: {error._result.result}")
        print("=> Landing and disarming for safety.")
        await drone.action.land()
        await asyncio.sleep(5)
        await drone.action.disarm()
        return
    except Exception as e:
        print(f"Unexpected error starting offboard: {e}")
        await drone.action.land()
        return

    # MOVE FORWARD
    print(f"Moving forward at {FORWARD_SPEED} m/s for {FORWARD_DURATION} s ...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_SPEED, 0.0, 0.0, 0.0))
    await asyncio.sleep(FORWARD_DURATION)

    # STOP
    print("Stopping motion (hover)...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(HOVER_AFTER_MOVE)

    # DROP PAYLOAD via servo
    print("Dropping payload (servo -> max)...")
    try:
        servo.min()   # move servo to drop position (adjust if needed)
    except Exception as e:
        print(f"Servo command failed: {e}")
    await asyncio.sleep(SERVO_DROP_TIME)

    # Optionally reset the servo to closed/ready position
    try:
        servo.max()
    except Exception:
        pass
    await asyncio.sleep(0.5)

    # MOVE BACK (return) — opposite forward velocity for roughly same time
    print(f"Moving back at {-FORWARD_SPEED} m/s for {FORWARD_DURATION} s ...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-FORWARD_SPEED, 0.0, 0.0, 0.0))
    await asyncio.sleep(FORWARD_DURATION)

    # STOP
    print("Stopping motion (hover) before landing...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(HOVER_AFTER_MOVE)

    # Stop offboard (return control to autopilot)
    try:
        print("Stopping offboard mode...")
        await drone.offboard.stop()
    except Exception as e:
        print(f"Warning: failed to stop offboard cleanly: {e}")

    # LAND
    print("Landing...")
    await drone.action.land()

    # give it a moment to land
    await asyncio.sleep(8)

    # disarm (optional; autopilot may disarm automatically once landed)
    try:
        print("Disarming...")
        await drone.action.disarm()
    except Exception:
        pass

    print("Mission complete — script exiting.")

if __name__ == "__main__":
    # Run the async sequence
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.")
        try:
            sys.exit(0)
        except SystemExit:
            pass
