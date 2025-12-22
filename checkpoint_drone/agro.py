#!/usr/bin/env python3

"""
NIDAR Obstacle Avoidance (NO ONNX MODEL)

Using:
    - Raspberry Pi 5
    - PiCam
    - Cube Orange (PX4/ArduPilot)
    - MAVSDK Offboard

Flow:
    - Capture frames from PiCam
    - Use classical CV (edges + dilation) as a fake "obstacle mask"
    - Build occupancy grid and detect obstacle ahead + free-space direction
    - State machine decides velocities in path frame
    - Convert to NED and send via MAVSDK Offboard
"""

import asyncio
import math
import time
from enum import Enum, auto

import cv2
import numpy as np
from picamera2 import Picamera2

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw


# =============================
# Perception (CV-based, no ONNX)
# =============================

class FreeSpaceDirection(Enum):
    NONE = auto()
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class Perception:
    def __init__(self,
                 input_size=(320, 240),
                 occ_grid_size=(40, 30),
                 obstacle_thresh=0.3):
        """
        CV-based "segmentation":
        - Resize frame
        - Convert to gray
        - Canny edges + dilation
        - Normalize to [0,1] as pseudo obstacle mask

        :param input_size: (width, height) for internal mask processing.
        :param occ_grid_size: (width, height) of occupancy grid.
        :param obstacle_thresh: threshold on edge magnitude to mark as obstacle.
        """
        self.model_w, self.model_h = input_size
        self.occ_w, self.occ_h = occ_grid_size
        self.obstacle_thresh = obstacle_thresh
        print("[Perception] Using classical CV (edges) instead of ONNX model")

    def infer_obstacle_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Simulate an obstacle mask using edges:
        - Gray -> Canny -> Dilate -> normalize to [0,1].
        :return: mask (H, W) float in [0,1].
        """
        resized = cv2.resize(frame_bgr, (self.model_w, self.model_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Canny edges (tune thresholds as needed)
        edges = cv2.Canny(gray, 80, 160)

        # Dilate edges to make them more "obstacle-like"
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        mask = edges_dilated.astype(np.float32) / 255.0  # [0,1]
        return mask

    def build_occupancy_grid(self, mask: np.ndarray):
        """
        Convert continuous mask into coarse occupancy grid.

        :param mask: float mask (H,W) in [0,1], 1 ~ obstacle-ish.
        :return: occ_grid (h,w) uint8 0/1
        """
        H, W = mask.shape
        # Use bottom half as near-ground region
        mask_near = mask[int(H * 0.5):, :]
        occ_grid = cv2.resize(mask_near, (self.occ_w, self.occ_h),
                              interpolation=cv2.INTER_NEAREST)
        occ_grid_bin = (occ_grid > self.obstacle_thresh).astype(np.uint8)
        return occ_grid_bin

    def analyze_free_space(self, occ_grid: np.ndarray,
                           bottom_rows=5,
                           min_free_pixels=10):
        """
        :param occ_grid: (h,w) binary (1=obstacle).
        :return: (obstacle_ahead (bool),
                  free_dir (FreeSpaceDirection),
                  free_centroid (x_norm,y_norm or None),
                  occ_debug (dict of counters))
        """
        h, w = occ_grid.shape
        # Region near bottom (closest to drone)
        danger_band = occ_grid[max(0, h - bottom_rows):, :]

        # Central danger window (bottom-center area)
        c_start = int(w * 0.33)
        c_end = int(w * 0.66)
        center_window = danger_band[:, c_start:c_end]
        center_obstacles = int(center_window.sum())
        obstacle_ahead = center_obstacles > (bottom_rows * (c_end - c_start) * 0.15)

        # Left / center / right counts (for debug)
        third = w // 3
        left_obs = int(danger_band[:, :third].sum())
        center_obs = int(danger_band[:, third:2 * third].sum())
        right_obs = int(danger_band[:, 2 * third:].sum())

        # Connected components on FREE cells to find largest free space touching bottom
        free = (occ_grid == 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            free, connectivity=8
        )

        best_area = 0
        best_centroid = None
        bottom_row_idx = h - 1

        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            # Only consider components that touch the bottom row (reachable free space)
            if bottom_row_idx not in ys:
                continue
            area = stats[label, cv2.CC_STAT_AREA]
            if area > best_area and area >= min_free_pixels:
                best_area = area
                cx, cy = centroids[label]
                best_centroid = (cx, cy)

        free_dir = FreeSpaceDirection.NONE
        free_centroid_norm = None

        if best_centroid is not None:
            cx, cy = best_centroid
            free_centroid_norm = (cx / w, cy / h)
            if cx < w / 3:
                free_dir = FreeSpaceDirection.LEFT
            elif cx > 2 * w / 3:
                free_dir = FreeSpaceDirection.RIGHT
            else:
                free_dir = FreeSpaceDirection.CENTER

        debug = {
            "left_obs": left_obs,
            "center_obs": center_obs,
            "right_obs": right_obs
        }

        return obstacle_ahead, free_dir, free_centroid_norm, debug


# =============================
# Path management
# =============================

class PathManager:
    def __init__(self):
        self.initialized = False
        self.N0 = 0.0
        self.E0 = 0.0
        self.psi0 = 0.0  # radians
        self.d = (1.0, 0.0)
        self.n = (0.0, 1.0)

    def init_from_pose(self, north_m: float, east_m: float, yaw_rad: float):
        self.N0 = north_m
        self.E0 = east_m
        self.psi0 = yaw_rad
        self.d = (math.cos(yaw_rad), math.sin(yaw_rad))
        self.n = (-math.sin(yaw_rad), math.cos(yaw_rad))
        self.initialized = True
        print(f"[Path] Initialized at N={self.N0:.2f}, E={self.E0:.2f}, yaw={math.degrees(yaw_rad):.1f} deg")

    def compute_errors(self, north_m: float, east_m: float):
        if not self.initialized:
            return 0.0, 0.0
        dpN = north_m - self.N0
        dpE = east_m - self.E0
        s = dpN * self.d[0] + dpE * self.d[1]  # along-track
        e = dpN * self.n[0] + dpE * self.n[1]  # cross-track
        return s, e


# =============================
# State machine & controller
# =============================

class AvoidState(Enum):
    FOLLOW_LINE = auto()
    AVOID_LEFT = auto()
    AVOID_RIGHT = auto()
    PASS_OBSTACLE = auto()
    RETURN_TO_LINE = auto()
    SAFE_HOLD = auto()


class AvoidanceController:
    def __init__(self,
                 path_manager: PathManager,
                 v_cruise=1.0,
                 v_avoid_fwd=0.6,
                 v_lat=0.8,
                 k_line=0.3,
                 k_return=0.4,
                 e_target=3.0,
                 e_tol=0.3):
        self.path = path_manager
        self.state = AvoidState.FOLLOW_LINE

        self.v_cruise = v_cruise
        self.v_avoid_fwd = v_avoid_fwd
        self.v_lat = v_lat
        self.k_line = k_line
        self.k_return = k_return
        self.e_target = e_target
        self.e_tol = e_tol

        self.pass_clear_frames = 0
        self.required_clear_frames = 10  # frames without obstacle

    def update(self,
               s: float,
               e: float,
               obstacle_ahead: bool,
               free_dir: FreeSpaceDirection) -> (float, float):
        """
        :return: (vx, vy) in path-aligned coordinates (x=forward, y=left->right).
        """
        vx = 0.0
        vy = 0.0

        # Safety: if perception says obstacle but no clear free direction
        if free_dir == FreeSpaceDirection.NONE and obstacle_ahead:
            self.state = AvoidState.SAFE_HOLD
        elif self.state == AvoidState.SAFE_HOLD and not obstacle_ahead:
            self.state = AvoidState.FOLLOW_LINE

        if self.state == AvoidState.SAFE_HOLD:
            return 0.0, 0.0

        # Transitions
        if self.state == AvoidState.FOLLOW_LINE:
            if obstacle_ahead:
                if free_dir == FreeSpaceDirection.LEFT:
                    self.state = AvoidState.AVOID_LEFT
                elif free_dir == FreeSpaceDirection.RIGHT:
                    self.state = AvoidState.AVOID_RIGHT
                else:
                    self.state = AvoidState.SAFE_HOLD

        elif self.state in (AvoidState.AVOID_LEFT, AvoidState.AVOID_RIGHT):
            # Start passing once we are laterally far enough from original line
            if abs(e) >= self.e_target:
                self.state = AvoidState.PASS_OBSTACLE
                self.pass_clear_frames = 0

        elif self.state == AvoidState.PASS_OBSTACLE:
            if not obstacle_ahead:
                self.pass_clear_frames += 1
            else:
                self.pass_clear_frames = 0

            if self.pass_clear_frames >= self.required_clear_frames:
                self.state = AvoidState.RETURN_TO_LINE

        elif self.state == AvoidState.RETURN_TO_LINE:
            if abs(e) < self.e_tol and not obstacle_ahead:
                self.state = AvoidState.FOLLOW_LINE

        # Actions
        if self.state == AvoidState.FOLLOW_LINE:
            vx = self.v_cruise
            vy = -self.k_line * e

        elif self.state == AvoidState.AVOID_LEFT:
            vx = self.v_avoid_fwd
            vy = -self.v_lat  # left = negative cross-track

        elif self.state == AvoidState.AVOID_RIGHT:
            vx = self.v_avoid_fwd
            vy = self.v_lat

        elif self.state == AvoidState.PASS_OBSTACLE:
            vx = self.v_cruise
            vy = -self.k_line * e

        elif self.state == AvoidState.RETURN_TO_LINE:
            vx = self.v_cruise
            vy = -self.k_return * e

        # Clamp
        vx = max(min(vx, 2.0), -2.0)
        vy = max(min(vy, 2.0), -2.0)
        return vx, vy


# =============================
# MAVSDK helper routines
# =============================

async def connect_drone(system_address: str = "serial:///dev/ttyACM0:921600") -> System:
    drone = System()
    await drone.connect(system_address=system_address)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break


    return drone


async def arm_and_takeoff(drone: System, altitude=5.0):
    print("-- Arming")
    await drone.action.arm()

    print(f"-- Taking off to {altitude} m")
    await drone.action.set_takeoff_altitude(altitude)
    await drone.action.takeoff()

    await asyncio.sleep(10.0)


async def start_offboard(drone: System, path: PathManager) -> float:
    """
    Initialize offboard with zero velocity, record initial yaw.
    :return: initial yaw in radians.
    """
    async for euler in drone.telemetry.attitude_euler():
        yaw_deg = euler.yaw_deg
        break

    yaw_rad = math.radians(yaw_deg)

    # Need current NED position
    async for pv in drone.telemetry.position_velocity_ned():
        north = pv.position.north_m
        east = pv.position.east_m
        break

    path.init_from_pose(north, east, yaw_rad)

    print("-- Initializing offboard")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw_deg))

    try:
        await drone.offboard.start()
        print("-- Offboard started")
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        await drone.action.disarm()
        raise

    return yaw_rad


# =============================
# Main loop
# =============================

async def main():
    # --- Initialize camera ---
    picam = Picamera2()
    cam_config = picam.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam.configure(cam_config)
    picam.start()
    time.sleep(2.0)

    # --- Initialize perception, path, controller ---
    perception = Perception(
        input_size=(320, 240),
        occ_grid_size=(40, 30),
        obstacle_thresh=0.3
    )
    path = PathManager()
    controller = AvoidanceController(path)

    # --- Connect to drone ---
    drone = await connect_drone("serial:///dev/ttyACM0:921600")  # adjust if needed
    await arm_and_takeoff(drone, altitude=5.0)
    yaw0 = await start_offboard(drone, path)
    yaw0_deg = math.degrees(yaw0)

    print("-- Starting obstacle avoidance loop (CV-based, no ONNX)")
    control_rate_hz = 15.0
    dt = 1.0 / control_rate_hz

    try:
        while True:
            loop_start = time.time()

            # 1. Grab frame
            frame = picam.capture_array()  # RGB888
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Perception
            mask = perception.infer_obstacle_mask(frame_bgr)
            occ_grid = perception.build_occupancy_grid(mask)
            obstacle_ahead, free_dir, free_centroid, debug = perception.analyze_free_space(
                occ_grid, bottom_rows=5, min_free_pixels=8
            )

            # 3. Get current pose in NED
            pv = await drone.telemetry.position_velocity_ned().__anext__()
            north = pv.position.north_m
            east = pv.position.east_m

            s, e = path.compute_errors(north, east)

            # 4. State machine -> path-frame velocities
            vx_path, vy_path = controller.update(s, e, obstacle_ahead, free_dir)

            # 5. Convert path-frame to NED-frame
            cpsi = path.d[0]
            spsi = path.d[1]
            vN = vx_path * cpsi + vy_path * (-spsi)
            vE = vx_path * spsi + vy_path * (cpsi)
            vD = 0.0  # Let autopilot hold altitude

            # 6. Send velocity command
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vN, vE, vD, yaw0_deg)
            )

            # 7. Debug print
            print(
                f"State={controller.state.name} "
                f"e={e:.2f}m obs={obstacle_ahead} free={free_dir.name} "
                f"vN={vN:.2f} vE={vE:.2f}"
            )

            # Timing
            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping offboard...")
    finally:
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()
        picam.stop()


if __name__ == "__main__":
    asyncio.run(main())
    
