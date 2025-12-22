# pos_mode_sequence.py
import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to vehicle")
            break

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(6)  # allow time to climb ~5m

    print("Starting manual control in Position mode...")
    await drone.manual_control.start_position_control()

    # --------------------------
    # Yaw right (~90°)
    print("Yawing right 90°...")
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.5)  # throttle=0.5 hold, yaw=0.5
    await asyncio.sleep(3)  # adjust time to get ~90°
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)

    # Pitch forward
    print("Pitching forward 5s...")
    await drone.manual_control.set_manual_control_input(0.5, 0.0, 0.5, 0.0)
    await asyncio.sleep(5)
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)

    # Pitch backward
    print("Pitching backward 5s...")
    await drone.manual_control.set_manual_control_input(-0.5, 0.0, 0.5, 0.0)
    await asyncio.sleep(5)
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)

    # Yaw another 90° (total 180° from start)
    print("Yawing right another 90°...")
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.5)
    await asyncio.sleep(3)
    await drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)

    # Land
    print("Landing...")
    await drone.action.land()
    await asyncio.sleep(10)
    print("Mission complete.")

if __name__ == "__main__":
    asyncio.run(run())