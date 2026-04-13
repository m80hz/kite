import argparse, os, json
from .config import PipelineConfig
from .pipeline import KitePipeline

def main():
    p = argparse.ArgumentParser(description='KITE')
    p.add_argument('--dataset_folder', type=str, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--test_file', type=str)
    group.add_argument('--test_dir', type=str)
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--model_url', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./outputs/kite')
    p.add_argument('--enable_llm_eval', action='store_true')
    p.add_argument('--llm_url', type=str, default=None)
    p.add_argument('--llm_model_name', type=str, default=None)
    p.add_argument('--robot_profile', type=str, default=None)
    p.add_argument('--yolo_weights', type=str, default=None)
    p.add_argument('--ovd_backend', type=str, default='auto', choices=['auto','owlvit','groundingdino','yolo','stub'])
    p.add_argument('--enable_3d_graph', action='store_true')
    p.add_argument('--enable_final_narrative', action='store_true')
    p.add_argument('--disable_tatc', action='store_true')
    p.add_argument('--dump_htatc', action='store_true', help='Dump per-QA H-TATC to htatc_dump.jsonl')
    p.add_argument('--ablate', type=str, default='', help='Comma-separated groups to disable: PLAN,SCENE3D,EVENTS,ROBOT,CONTACT,BIMANUAL')
    p.add_argument('--force_bimanual_tokens', action='store_true')
    args = p.parse_args()

    cfg = PipelineConfig()
    cfg.data.dataset_folder = args.dataset_folder
    cfg.model.model_name = args.model_name
    cfg.model.model_url = args.model_url

    pipe = KitePipeline(cfg)

    ablate = [s.strip() for s in args.ablate.split(',') if s.strip()] if args.ablate else []

    if args.test_file:
        stats = pipe.run_full_eval_file(
            test_file=args.test_file,
            out_dir=args.out_dir,
            enable_llm_eval=args.enable_llm_eval,
            llm_url=args.llm_url,
            llm_model_name=args.llm_model_name,
            robot_profile=args.robot_profile,
            yolo_weights=args.yolo_weights,
            ovd_backend=args.ovd_backend,
            enable_3d_graph=args.enable_3d_graph,
            enable_final_narrative=args.enable_final_narrative,
            enable_tatc=(not args.disable_tatc),
            dump_htatc=args.dump_htatc,
            ablate=ablate,
            force_bimanual_tokens=args.force_bimanual_tokens
        )
        print(json.dumps({'split_stats_written_to': os.path.join(args.out_dir, 'stats_data.json')}, indent=2))
    else:
        merged = pipe.run_full_eval_dir(
            test_dir=args.test_dir,
            out_root=args.out_dir,
            enable_llm_eval=args.enable_llm_eval,
            llm_url=args.llm_url,
            llm_model_name=args.llm_model_name,
            robot_profile=args.robot_profile,
            yolo_weights=args.yolo_weights,
            ovd_backend=args.ovd_backend,
            enable_3d_graph=args.enable_3d_graph,
            enable_final_narrative=args.enable_final_narrative,
            enable_tatc=(not args.disable_tatc),
            dump_htatc=args.dump_htatc,
            ablate=ablate,
            force_bimanual_tokens=args.force_bimanual_tokens
        )
        print(json.dumps({'merged_results_written_to': os.path.join(args.out_dir, 'results_merged.json')}, indent=2))

if __name__ == '__main__':
    main()
