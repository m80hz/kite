from dataclasses import dataclass
from typing import Dict, Any, Optional
import json, os

@dataclass
class RobotProfile:
    name: str = "UnknownRobot"
    morphology: str = "single-arm"
    arms: int = 1
    grippers: int = 1
    end_effectors: str = "parallel_jaw"
    sensors: str = "rgb video"
    workspace: Optional[str] = None
    notes: Optional[str] = None

    @staticmethod
    def load(path: Optional[str]) -> "RobotProfile":
        if path is None or not os.path.exists(path):
            return RobotProfile()
        with open(path, "r") as f:
            d = json.load(f)
        # Only keep keys that correspond to dataclass fields
        valid_keys = set(RobotProfile.__annotations__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Accept sensors as either a string or a list and normalize to a string
        if "sensors" in filtered and isinstance(filtered["sensors"], (list, tuple)):
            filtered["sensors"] = ", ".join(map(str, filtered["sensors"]))
        return RobotProfile(**filtered)
    
    def as_prompt_text(self) -> str:
        # Normalize sensors for display (accepts str or list)
        sensors = self.sensors
        if isinstance(sensors, (list, tuple)):
            sensors = ", ".join(map(str, sensors))

        parts = [
            f"Robot name: {self.name}.",
            f"Morphology: {self.morphology}.",
            f"Arms: {self.arms}. Grippers: {self.grippers}. End-effector: {self.end_effectors}.",
            f"Sensors: {sensors}.",
        ]
        if self.workspace: parts.append(f"Workspace: {self.workspace}.")
        if self.notes: parts.append(f"Notes: {self.notes}.")
        return " ".join(parts)
