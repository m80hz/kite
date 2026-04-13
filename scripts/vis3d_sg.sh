# !/bin/bash
# Visualize local 3D scene graph from a video frame using Depth-Anything-V2 and OV-D (e.g., OWL-ViT)

# Example usage:
python tools/vis3d_local_sg.py \
       --video_path datasets/robofac/simulation_data/MicrowaveTask-mug/spoon_grasp_view2/8c14634c-1ed3-42f7-b6bd-f0b02a8292ae.mp4 \
       --time_sec 2.5 \
       --ovd_backend owlvit \
       --save_ply ./outputs/vis/8c146_local3d.ply \
       --save_png ./outputs/vis/8c146_local3d.png \
       --save_depth \
       --save_rgb \
       --show




