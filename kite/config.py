from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name: str = "qwen2.5-vl-7b-robofac"
    model_url: str = "http://127.0.0.1:8000/v1"

@dataclass
class DataConfig:
    dataset_folder: str = "./datasets/robofac/simulation_data"
    test_file: str = "./datasets/robofac/test_qa_sim/annos_per_video_split0.json"
    frame_interval: int = 30  # frames to skip when extracting images for prompts

@dataclass
class SegmentConfig:
    max_segments: int = 8
    window_sec: float = 2.0
    overlap_sec: float = 0.5
    use_scene_cuts: bool = True
    flow_stride: int = 2
    min_flow_mag: float = 0.5

@dataclass
class RetrievalConfig:
    enable: bool = True
    index_path: Optional[str] = None
    topk: int = 8
    lambda_prior: float = 0.5  # blend weight for prior in log-odds

@dataclass
class NarrationConfig:
    mode: str = "info"  # "alert" | "info" | "debug"
    include_evidence: bool = True
    two_voice: bool = True

@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
