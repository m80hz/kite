from kite.config import PipelineConfig
from kite.pipeline import KitePipeline

if __name__ == '__main__':
    cfg = PipelineConfig()
    # Fill these before running:
    cfg.data.dataset_folder = 'datasets/robofac/simulation_data'
    cfg.data.test_file = 'datasets/robofac/test_qa_sim/dummy.json'
    cfg.model.model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    cfg.model.model_url = 'http://127.0.0.1:8000/v1'
    pipe = KitePipeline(cfg)

    stats = pipe.run_full_eval_file(cfg.data.test_file, './outputs/demo')
    print('split_stats:', stats)
