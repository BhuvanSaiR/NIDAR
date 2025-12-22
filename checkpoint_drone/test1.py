# yaw_mavsdk_mission.py
import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyACM0:115200")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to vehicle")
            break

    # --- Parameters you can tune ---
    TAKEOFF_ALT = 5.0       # meters
    FORWARD_VEL = 1.0       # m/s for 'pitch forward' (body-frame forward)
    YAW_SETTLE = 4.0        # seconds to let yaw reach target
    PITCH_DURATION = 5.0    # seconds forward/back

    # try to set desired takeoff altitude (optional — may fail on some firmwares)
    try:
        await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
    except Exception as e:
        print("Note: set_takeoff_altitude not supported / failed:", e)

    print("Arming...")
    await drone.action.arm()

    print(f"Taking off to {TAKEOFF_ALT} m...")
    await drone.action.takeoff()

    # wait until roughly at target altitude
    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        print(f"Altitude: {alt:.2f} m")
        if alt >= (TAKEOFF_ALT - 0.5):
            print("Reached takeoff altitude")
            break
        await asyncio.sleep(0.2)

    # Prepare and start offboard: must send an initial setpoint before start
    initial_yaw = 0.0  # degrees (0 = north)
    print("Sending initial offboard setpoint and starting offboard...")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, initial_yaw))
    await drone.offboard.start()

    try:
        # --- Yaw to 90 degrees (absolute) ---
        print("Yaw -> 90° (absolute)")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, 90.0))
        await asyncio.sleep(YAW_SETTLE)

        # --- Pitch forward (body-frame forward velocity) ---
        print(f"Pitch forward for {PITCH_DURATION}s at {FORWARD_VEL} m/s")
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(FORWARD_VEL, 0.0, 0.0, 0.0))
        await asyncio.sleep(PITCH_DURATION)

        # --- Pitch backward (reverse forward velocity) ---
        print(f"Pitch backward for {PITCH_DURATION}s at {-FORWARD_VEL} m/s")
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(-FORWARD_VEL, 0.0, 0.0, 0.0))
        await asyncio.sleep(PITCH_DURATION)

        # Stop body motion
        print("Stopping body motion")
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(1.0)

        # --- Yaw another 90 degrees (to 180° absolute) ---
        print("Yaw -> 180° (absolute)")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, 180.0))
        await asyncio.sleep(YAW_SETTLE)

        # Ensure we are holding the original XY takeoff point (0,0) before landing
        print("Holding takeoff position (0,0) for 2s")
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -TAKEOFF_ALT, 180.0))
        await asyncio.sleep(2.0)

    finally:
        # always try to stop offboard before landing
        print("Stopping offboard and landing...")
        try:
            await drone.offboard.stop()
        except Exception as e:
            print("Warning: offboard.stop() failed:", e)

        await drone.action.land()
        print("Landing command issued. Mission complete.")

if __name__ == "__main__":
    asyncio.run(run())