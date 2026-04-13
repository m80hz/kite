#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Keeps coordinate/back-projection logic IDENTICAL to the original (X right, Y down, Z forward).
- Keeps the Open-Vocab detection & 3D scene graph logic intact.
- Supports: saving PLY (point cloud), saving PNG (offscreen), interactive viewing (--show).
- Draws nodes as spheres and edges as lines or world-space cylinders for "thick" edges.

Usage example:
    python vis3d_local_sg_pyvista.py --video_path demo.mp4 --time_sec 2.0 --save_png out.png --show
"""

import os, argparse, numpy as np, cv2, hashlib, colorsys
import math

# --- Preferred visualization backends ---
HAS_PYVISTA = False
HAS_VEDO = False
try:
    import pyvista as pv
    HAS_PYVISTA = True
except Exception:
    try:
        import vedo as vd
        HAS_VEDO = True
    except Exception:
        pass

# --- Your perception & scene-graph stack (unchanged) ---
from kite.perception.camera import CameraIntrinsics
from kite.perception.depth3d import DepthEstimator
from kite.perception.detector_openvocab import OpenVocabDetector
from kite.perception.scene_graph3d import build_local_graph3d


# ---------------------------
# Utility helpers (kept compatible with original)
# ---------------------------

def color_from_name(name: str):
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h/360.0, 0.6, 0.95)
    return (int(255*r), int(255*g), int(255*b))


def frame_to_pointcloud(rgb, depth, cam, z_scale=1.0):
    """Unchanged back-projection logic from original script."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth * z_scale
    X = (us - cam.cx) * Z / cam.fx
    Y = (vs - cam.cy) * Z / cam.fy
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    # Original color handling: convert BGR->RGB and normalize to [0,1] for renderers
    cols = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0
    valid = Z.reshape(-1) > 1e-6
    return pts[valid], cols[valid]


def write_ply_pointcloud(path: str, pts: np.ndarray, cols01: np.ndarray) -> None:
    """Simple ASCII PLY writer (point cloud with vertex colors)."""
    cols255 = np.clip(cols01 * 255.0, 0, 255).astype(np.uint8)
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, cols255):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


# ---------------------------
# Visualization builders
# ---------------------------

def _compute_scene_bounds(point_arrays: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Return (center, extent) using an AABB over provided Nx3 arrays."""
    pts_all = [p for p in point_arrays if p is not None and len(p) > 0]
    if not pts_all:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0
    cat = np.concatenate(pts_all, axis=0)
    mn = np.min(cat, axis=0)
    mx = np.max(cat, axis=0)
    center = (mn + mx) * 0.5
    extent = float(np.max(mx - mn)) if np.any(mx > mn) else 1.0
    return center, extent


def _make_cylinder_pyvista(p0: np.ndarray, p1: np.ndarray, radius: float, color_rgb01: tuple[float, float, float]):
    v = (p1 - p0).astype(np.float64)
    L = float(np.linalg.norm(v))
    if L <= 1e-9:
        return None
    direction = v / L
    center = (p0 + p1) * 0.5
    cyl = pv.Cylinder(center=center, direction=direction, radius=float(radius), height=L, resolution=24)
    return cyl, color_rgb01


def visualize_with_pyvista(pts: np.ndarray,
                           cols01: np.ndarray,
                           node_centers: list[list[float]],
                           edge_pairs: list[tuple[int, int]],
                           edge_colors01: list[tuple[float, float, float]],
                           line_thickness: float,
                           save_png: str | None,
                           show_interactive: bool) -> None:
    off = not show_interactive
    plotter = pv.Plotter(off_screen=off, window_size=(1280, 720))
    plotter.set_background('white')
    plotter.add_axes()

    # Point cloud
    if len(pts) > 0:
        pc = pv.PolyData(pts)
        pc['RGB'] = (cols01 * 255.0).astype(np.uint8)  # PyVista supports uint8 RGB
        plotter.add_mesh(pc, scalars='RGB', rgb=True, point_size=2.0, render_points_as_spheres=True, name='pointcloud')

    # Nodes (Octahedron at centroids)
    spheres_pts = []
    for n in node_centers:
        # sph = pv.Sphere(radius=0.05, center=n)
        # sph = pv.Octahedron(radius=0.05, center=n)
        sph = pv.Icosahedron(radius=0.05, center=n)
        plotter.add_mesh(sph, color='white', specular=0.2, name=f"node_{len(spheres_pts)}")
        spheres_pts.append(np.asarray(n, dtype=np.float64).reshape(1, 3))

    # Edges: lines or cylinders
    if edge_pairs:
        if line_thickness and line_thickness > 0.0:
            # cylinders in world units
            for (a, b), col in zip(edge_pairs, edge_colors01):
                p0 = np.array(node_centers[a], dtype=np.float64)
                p1 = np.array(node_centers[b], dtype=np.float64)
                made = _make_cylinder_pyvista(p0, p1, radius=float(line_thickness), color_rgb01=col)
                if made is None:
                    continue
                cyl, color = made
                plotter.add_mesh(cyl, color=color, specular=0.2)
        else:
            # simple line segments (screen-space width)
            for (a, b), col in zip(edge_pairs, edge_colors01):
                p0 = np.array(node_centers[a], dtype=np.float64)
                p1 = np.array(node_centers[b], dtype=np.float64)
                line = pv.Line(p0, p1)
                plotter.add_mesh(line, color=col, line_width=3)

    # Camera setup similar to original intent
    center, extent = _compute_scene_bounds([pts] + spheres_pts if spheres_pts else [pts])
    eye = center - np.array([0.0, 0.0, 3.0 * extent], dtype=np.float64)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # match original "up" to avoid behavior change
    plotter.camera_position = [tuple(eye), tuple(center), tuple(up)]

    # Save screenshot and/or show
    if save_png:
        plotter.show(screenshot=save_png, auto_close=not show_interactive)
        if show_interactive:
            # If we also want interaction, re-open without auto-close
            plotter.show()
    elif show_interactive:
        plotter.show()


def visualize_with_vedo(pts: np.ndarray,
                        cols01: np.ndarray,
                        node_centers: list[list[float]],
                        edge_pairs: list[tuple[int, int]],
                        edge_colors01: list[tuple[float, float, float]],
                        line_thickness: float,
                        save_png: str | None,
                        show_interactive: bool) -> None:
    vd.settings.default_backend = 'vtk'
    actors = []

    if len(pts) > 0:
        pc = vd.Points(pts, r=3)
        pc.pointColors(cols01)  # expects 0..1
        actors.append(pc)

    # Nodes
    for n in node_centers:
        sph = vd.Sphere(pos=tuple(n), r=0.01, c='white')
        actors.append(sph)

    # Edges
    if edge_pairs:
        if line_thickness and line_thickness > 0.0:
            for (a, b), col in zip(edge_pairs, edge_colors01):
                p0 = node_centers[a]
                p1 = node_centers[b]
                cyl = vd.Cylinder(p0, p1, r=float(line_thickness))
                cyl.c(col)
                actors.append(cyl)
        else:
            for (a, b), col in zip(edge_pairs, edge_colors01):
                p0 = node_centers[a]
                p1 = node_centers[b]
                ln = vd.Line(p0, p1).lw(3).c(col)
                actors.append(ln)

    plt = vd.Plotter(size=(1280, 720), bg='white')
    plt.show(actors, interactive=False)
    if save_png:
        plt.screenshot(save_png)
    if show_interactive:
        plt.interactive().close()
    else:
        plt.close()


# ---------------------------
# Main (unchanged logic for I/O, depth, detections, scene graph)
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_path', type=str, required=True)
    ap.add_argument('--time_sec', type=float, default=0.0)
    ap.add_argument('--frame_idx', type=int, default=None)
    ap.add_argument('--ovd_backend', type=str, default='auto')
    ap.add_argument('--yolo_weights', type=str, default=None)
    ap.add_argument('--thirdparty_root', type=str, default='thirdparty/Depth-Anything-V2')
    ap.add_argument('--z_scale', type=float, default=1.0)
    ap.add_argument('--pred_is_inverse', action='store_true', help='If set, treat network prediction as inverse depth and invert it')
    ap.add_argument('--no_pred_is_inverse', dest='pred_is_inverse', action='store_false')
    ap.set_defaults(pred_is_inverse=True)
    ap.add_argument('--scale_pred', type=float, default=1000.0, help='Scale used when converting predictions to depth (numerator when inverting)')
    ap.add_argument('--min_depth', type=float, default=0.01, help='Minimum depth (meters) to clip to after conversion')
    ap.add_argument('--max_depth', type=float, default=10.0, help='Maximum depth (meters) to clip to after conversion')
    ap.add_argument('--save_ply', type=str, default=None)
    ap.add_argument('--save_png', type=str, default=None)
    ap.add_argument('--save_depth', action='store_true', help='Also save depth image (16-bit PNG) in same output dir')
    ap.add_argument('--save_rgb', action='store_true', help='Also save RGB frame image in same output dir')
    ap.add_argument('--line_thickness', type=float, default=0.0, help='If >0, draw edges as thin cylinders with this radius (meters)')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.frame_idx is not None:
        f = args.frame_idx
    else:
        f = int((args.time_sec or 0.0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f))
    ok, fr = cap.read(); cap.release()
    if not ok:
        raise RuntimeError('Could not read frame at requested time/index')
    H, W = fr.shape[:2]
    cam = CameraIntrinsics.from_image_size(W, H, fov_deg=180.0/(math.pi))

    depth_est = DepthEstimator(thirdparty_root=args.thirdparty_root,
                               pred_is_inverse=bool(args.pred_is_inverse),
                               scale_pred=float(args.scale_pred),
                               min_depth=float(args.min_depth),
                               max_depth=float(args.max_depth))
    depth = depth_est.predict(fr)
    print(f"[INFO] Estimated depth: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")
    print(depth)

    # Optionally save depth and RGB to the same output directory (unchanged logic)
    if args.save_depth or args.save_rgb:
        out_dir = None
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        if args.save_ply:
            out_dir = os.path.dirname(args.save_ply) or '.'
            base_name = os.path.splitext(os.path.basename(args.save_ply))[0]
        elif args.save_png:
            out_dir = os.path.dirname(args.save_png) or '.'
            base_name = os.path.splitext(os.path.basename(args.save_png))[0]
        else:
            out_dir = os.path.join('.', 'outputs', 'vis')
        os.makedirs(out_dir, exist_ok=True)
        # normalize depth for saving (0..1) — same behavior as original
        try:
            mn = float(depth.min()); mx = float(depth.max())
            if mx - mn > 1e-6:
                depth_norm = (depth - mn) / (mx - mn)
            else:
                depth_norm = depth * 0.0
        except Exception:
            depth_norm = depth

        if args.save_depth:
            depth_u16 = (depth_norm * 65535.0).astype(np.uint16)
            depth_path = os.path.join(out_dir, f"{base_name}_depth.png")
            cv2.imwrite(depth_path, depth_u16)
            print('[OK] wrote', depth_path)

        if args.save_rgb:
            rgb_path = os.path.join(out_dir, f"{base_name}_rgb.png")
            cv2.imwrite(rgb_path, fr)
            print('[OK] wrote', rgb_path)

    # Detections & local scene-graph (unchanged)
    ovdet = OpenVocabDetector(backend=args.ovd_backend, yolo_weights=args.yolo_weights)
    dets = ovdet.detect(fr, text_queries=['robot arm .','gripper .','end effector .','tool .','cup .','mug .','bottle .','drawer .','door .','microwave .','cube .','object .'])

    g_local = build_local_graph3d(dets, depth, cam, t_sec=(f/fps), frame_idx=f)

    # Build geometry: point cloud
    pts, cols = frame_to_pointcloud(fr, depth, cam, z_scale=args.z_scale)

    # Save PLY if requested (point cloud only, same semantics as original)
    if args.save_ply:
        try:
            write_ply_pointcloud(args.save_ply, pts, cols)
            print('[OK] wrote', args.save_ply)
        except Exception as e:
            print('[WARN] Failed writing PLY:', e)

    # Collect node centers & edge list/colors (unchanged semantics)
    node_centers = []
    for n in g_local.nodes:
        cx, cy, cz = n.point3d
        node_centers.append([float(cx), float(cy), float(cz)])

    edge_pairs = []
    edge_colors01 = []
    if len(g_local.edges) > 0 and len(node_centers) > 0:
        for e in g_local.edges:
            edge_pairs.append([e.subj, e.obj])
            # red for near/in_front_of, otherwise green (match original intent)
            edge_colors01.append((1.0, 0.0, 0.0) if e.pred in ('near', 'in_front_of') else (0.0, 1.0, 0.0))

    # Visualization & screenshot
    if not HAS_PYVISTA and not HAS_VEDO:
        if args.save_png or args.show:
            print('[WARN] No visualization backend installed (pyvista/vedo). Skipping rendering.')
    else:
        if HAS_PYVISTA:
            visualize_with_pyvista(pts=pts,
                                   cols01=cols,
                                   node_centers=node_centers,
                                   edge_pairs=edge_pairs,
                                   edge_colors01=edge_colors01,
                                   line_thickness=float(args.line_thickness),
                                   save_png=args.save_png,
                                   show_interactive=bool(args.show))
        else:
            visualize_with_vedo(pts=pts,
                                cols01=cols,
                                node_centers=node_centers,
                                edge_pairs=edge_pairs,
                                edge_colors01=edge_colors01,
                                line_thickness=float(args.line_thickness),
                                save_png=args.save_png,
                                show_interactive=bool(args.show))

    print('[DONE]')


if __name__ == '__main__':
    main()
