from .config import PipelineConfig
from .eval.full_eval import evaluate_split, evaluate_dir


class KitePipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run_full_eval_file(self, test_file: str, out_dir: str, enable_llm_eval: bool=False, llm_url: str=None, llm_model_name: str=None, robot_profile: str=None, yolo_weights: str=None, ovd_backend: str='auto', enable_3d_graph: bool=False, enable_final_narrative: bool=False, enable_tatc: bool=True, dump_htatc: bool=False, ablate=None, force_bimanual_tokens: bool=False):
        return evaluate_split(self.cfg.data.dataset_folder, test_file, out_dir, self.cfg.model.model_name, self.cfg.model.model_url, enable_llm_eval, llm_url, llm_model_name, robot_profile, yolo_weights, ovd_backend, enable_3d_graph, enable_final_narrative, enable_tatc, dump_htatc, ablate, force_bimanual_tokens)

    def run_full_eval_dir(self, test_dir: str, out_root: str, enable_llm_eval: bool=False, llm_url: str=None, llm_model_name: str=None, robot_profile: str=None, yolo_weights: str=None, ovd_backend: str='auto', enable_3d_graph: bool=False, enable_final_narrative: bool=False, enable_tatc: bool=True, dump_htatc: bool=False, ablate=None, force_bimanual_tokens: bool=False):
        return evaluate_dir(self.cfg.data.dataset_folder, test_dir, out_root, self.cfg.model.model_name, self.cfg.model.model_url, enable_llm_eval, llm_url, llm_model_name, robot_profile, yolo_weights, ovd_backend=ovd_backend, enable_3d_graph=enable_3d_graph, enable_final_narrative=enable_final_narrative, enable_tatc=enable_tatc, dump_htatc=dump_htatc, ablate=ablate, force_bimanual_tokens=force_bimanual_tokens)
