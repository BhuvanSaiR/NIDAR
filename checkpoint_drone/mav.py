# yaw_mavsdk.py
import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(5)  # give time to climb

    # IMPORTANT: set an initial offboard setpoint BEFORE starting offboard
    # NED: north=0, east=0, down=-2.0  (negative down => up 2 m). yaw in degrees.
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -2.0, 0.0))
    await drone.offboard.start()
    print("Offboard started — holding position.")

    # --- Method A: set absolute yaw to 90 degrees (east) while holding position ---
    print("Yaw -> 90° (absolute)...")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -2.0, 90.0))
    await asyncio.sleep(4)  # wait for rotation to complete

    # --- Method B: rotate at a yaw rate (30 deg/s clockwise) for 3 seconds ---
    print("Rotate at +30 deg/s for 3s...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
    await asyncio.sleep(3)
    # stop rotation by sending zero yaw rate
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(1)

    print("Stopping offboard and landing...")
    await drone.offboard.stop()
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())