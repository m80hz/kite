import itertools, subprocess, argparse, os

GROUPS = ['PLAN','SCENE3D','EVENTS','ROBOT','CONTACT','BIMANUAL']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cli', type=str, default='python -m kite.cli')
    ap.add_argument('--dataset_folder', type=str, required=True)
    ap.add_argument('--test_dir', type=str, required=True)
    ap.add_argument('--model_name', type=str, required=True)
    ap.add_argument('--model_url', type=str, required=True)
    ap.add_argument('--out_root', type=str, required=True)
    ap.add_argument('--robot_profile', type=str, default=None)
    ap.add_argument('--yolo_weights', type=str, default=None)
    ap.add_argument('--ovd_backend', type=str, default='auto')
    ap.add_argument('--enable_3d_graph', action='store_true')
    ap.add_argument('--enable_final_narrative', action='store_true')
    ap.add_argument('--grid', type=str, default='PLAN,SCENE3D,EVENTS,ROBOT,CONTACT,BIMANUAL', help='Comma-separated groups to toggle one-at-a-time')
    args = ap.parse_args()

    toggles = [g.strip().upper() for g in args.grid.split(',') if g.strip()]
    for g in toggles:
        out_dir = os.path.join(args.out_root, f"ablate_{g.lower()}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            *args.cli.split(' '),
            '--dataset_folder', args.dataset_folder,
            '--test_dir', args.test_dir,
            '--model_name', args.model_name,
            '--model_url', args.model_url,
            '--out_dir', out_dir,
            '--ablate', g,
            '--dump_htatc'
        ]
        if args.robot_profile: cmd += ['--robot_profile', args.robot_profile]
        if args.yolo_weights: cmd += ['--yolo_weights', args.yolo_weights]
        if args.ovd_backend:  cmd += ['--ovd_backend', args.ovd_backend]
        if args.enable_3d_graph: cmd += ['--enable_3d_graph']
        if args.enable_final_narrative: cmd += ['--enable_final_narrative']
        print('[RUN]', ' '.join(cmd))
        subprocess.run(cmd, check=False)

if __name__ == '__main__':
    main()
