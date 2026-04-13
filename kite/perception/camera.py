from dataclasses import dataclass

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    # default fov is based on typical fov of 1 radian in maniskill simulator
    @staticmethod
    def from_image_size(width: int, height: int, fov_deg: float = 57.32) -> "CameraIntrinsics":
        import math
        fx = (width/2) / math.tan((fov_deg/2) * math.pi/180.0)
        fy = fx
        cx = width/2
        cy = height/2
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
