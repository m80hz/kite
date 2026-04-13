from kite.video.optflow import compute_flow_magnitude_sequence
import numpy as np

mags, idxs = compute_flow_magnitude_sequence("datasets/robofac/realworld_data/so100_insert_cylinder_error/videos/chunk-000/observation.images.rightfront/episode_000000.mp4", stride=1, resize=(512,512), transcode_if_needed=True)
print("len:", len(mags))
if len(mags)>0:
    print("max:", np.max(mags), "mean:", np.mean(mags))
    print("p50/p90/p99:", np.percentile(mags, [50,90,99]))
else:
    print("no magnitudes extracted")


from kite.video.segmenter import segment_video
segs, fps = segment_video("datasets/robofac/realworld_data/so100_insert_cylinder_error/videos/chunk-000/observation.images.rightfront/episode_000000.mp4",
                          flow_stride=1,
                          window_sec=0.4,
                          overlap_sec=0.2,
                          max_segments=12,
                          min_flow_mag=0.01,
                          transcode_if_needed=True)
print("segments:", segs, "fps:", fps)