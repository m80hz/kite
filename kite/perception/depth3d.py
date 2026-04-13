import numpy as np

class DepthEstimator:
    def __init__(self,
                 thirdparty_root: str = "thirdparty/Depth-Anything-V2",
                 model_type: str = "vitl",
                 device: str = "cuda",
                 *,
                 pred_is_inverse: bool = True,
                 scale_pred: float = 1.0,
                 min_depth: float = 1e-3,
                 max_depth: float = 100.0,
                 eps: float = 1e-6,
                 # Outlier cropping (quantile-based)
                 enable_outlier_crop: bool = True,
                 crop_low_q: float = 0.0,
                 crop_high_q: float = 0.80,
                 crop_mode: str = "mask",  # 'clip' or 'mask'
                 # Room-scale normalization (map high quantile to target meters)
                 enable_room_scale: bool = True,
                 room_depth_q: float = 0.80,
                 room_max_depth_m: float = 4.0):
        """DepthEstimator wrapper for Depth-Anything-V2 model.

        Args:
            thirdparty_root: path to Depth-Anything-V2 thirdparty code.
            model_type: model encoder/type string passed to the model.
            device: device string for model (e.g., 'cuda' or 'cpu').
            pred_is_inverse: if True, the network output is treated as inverse-relative depth
                and will be inverted to produce depth: depth = scale_pred / (pred + eps).
            scale_pred: multiplicative scale applied after conversion (or used as numerator when inverting).
            min_depth, max_depth: optional absolute clipping bounds applied to final depth (post-processing).
            eps: small epsilon to avoid divide-by-zero when inverting.
            enable_outlier_crop: if True, crop depth outliers based on quantiles (after conversion and base clipping).
            crop_low_q, crop_high_q: quantile bounds in [0,1]; e.g., 0.0 and 0.99 to ignore extreme far values.
            crop_mode: 'clip' (saturate to bounds) or 'mask' (set outside-bounds to 0.0 so they are ignored later).
            enable_room_scale: if True, rescale depth so that the room_depth_q quantile maps to room_max_depth_m meters.
            room_depth_q: the quantile of depth used as target (e.g., 0.98 or 0.99).
            room_max_depth_m: the desired depth value (meters) for the chosen quantile; scales all depths proportionally.
        """
        self.thirdparty_root = thirdparty_root
        self.model_type = model_type
        self.device = device
        self.pred_is_inverse = bool(pred_is_inverse)
        self.scale_pred = float(scale_pred)
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None
        self.eps = float(eps)
        self.model = None
        # Outlier crop/scale settings
        self.enable_outlier_crop = bool(enable_outlier_crop)
        self.crop_low_q = float(crop_low_q)
        self.crop_high_q = float(crop_high_q)
        self.crop_mode = str(crop_mode).lower()
        self.enable_room_scale = bool(enable_room_scale)
        self.room_depth_q = float(room_depth_q)
        self.room_max_depth_m = float(room_max_depth_m)
        self._lazy_load()

    def _lazy_load(self):
        import sys, os
        tp = os.path.abspath(self.thirdparty_root)
        if tp not in sys.path:
            sys.path.insert(0, tp)
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            # DepthAnythingV2 expects encoder string as first arg; avoid passing device as positional
            self.model = DepthAnythingV2(self.model_type)
            # move model to requested device explicitly
            try:
                self.model = self.model.to(self.device)
            except Exception:
                # if device string is invalid or not supported, ignore and keep model on default device
                pass
            # try loading pretrained checkpoint if present (Depth-Anything-V2 expects it under checkpoints/)
            try:
                import torch
                ckpt = os.path.join(tp, 'checkpoints', f'depth_anything_v2_{self.model_type}.pth')
                # ckpt = os.path.join(tp, 'checkpoints', f'depth_anything_v2_metric_hypersim_{self.model_type}.pth')
                if os.path.isfile(ckpt):
                    state = torch.load(ckpt, map_location=self.device if isinstance(self.device, str) else 'cpu')
                    # allow strict=False in case keys differ slightly
                    missing, unexpected = self.model.load_state_dict(state, strict=False)
                    # optional: print a tiny note only when keys mismatch
                    if (missing or unexpected):
                        print(f"[DepthEstimator] Loaded weights with missing={len(missing)}, unexpected={len(unexpected)}")
                else:
                    print(f"[DepthEstimator] Checkpoint not found: {ckpt}. Running with randomly initialized weights (depth may be poor).")
                self.model.eval()
            except Exception as we:
                print(f"[DepthEstimator] Warning: failed to load weights: {we}")
        except Exception as e:
            # fallback: try build_model if available; surface clearer error if not
            try:
                from depth_anything_v2 import build_model
                self.model = build_model(self.model_type, device=self.device)
            except Exception as e2:
                raise ImportError(f"Failed to import or build DepthAnything model: {e}; fallback error: {e2}")

    def predict(self, frame_bgr) -> np.ndarray:
        # run model inference (DepthAnything returns a 2D numpy array)
        try:
            depth_pred = self.model.infer_image(frame_bgr)
        except Exception as e:
            # Common failures: CUDA out of memory or device mismatch between model buffers and input.
            msg = str(e).lower()
            if 'out of memory' in msg or 'device' in msg or 'cuda' in msg:
                print(f"[DepthEstimator] Inference failed ({e}); falling back to CPU and retrying.")
                try:
                    # move model to CPU and retry
                    import torch
                    self.model = self.model.to('cpu')
                except Exception:
                    pass
                depth_pred = self.model.infer_image(frame_bgr)
            else:
                # re-raise unexpected errors
                raise
        depth_pred = depth_pred.astype("float32")
        return self._process_prediction(depth_pred)

    def _process_prediction(self, pred: np.ndarray) -> np.ndarray:
        """Convert network prediction to depth.

        If the network produces inverse-relative depth (common for some models), set
        `pred_is_inverse=True` and the function returns:
            depth = scale_pred / (pred + eps)

        Otherwise it will multiply by `scale_pred`.
        """
        pred = pred.astype("float32")
        if self.pred_is_inverse:
            # avoid division by zero and negative values
            safe_pred = np.clip(pred, a_min=self.eps, a_max=None)
            depth = (self.scale_pred) / safe_pred
        else:
            depth = pred * (self.scale_pred)
        # ensure no non-finite numbers
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        # apply min/max clipping if provided
        if (self.min_depth is not None) or (self.max_depth is not None):
            mn = self.min_depth if self.min_depth is not None else np.min(depth)
            mx = self.max_depth if self.max_depth is not None else np.max(depth)
            if mx < mn:
                # guard: if user supplied inverted bounds, swap
                mn, mx = mx, mn
            depth = np.clip(depth, a_min=mn, a_max=mx)

        # Quantile-based outlier cropping (post base clipping)
        if self.enable_outlier_crop:
            valid = depth > 0.0
            if np.any(valid):
                try:
                    loq = np.clip(self.crop_low_q, 0.0, 1.0)
                    hiq = np.clip(self.crop_high_q, 0.0, 1.0)
                    if hiq < loq:
                        loq, hiq = hiq, loq
                    vals = depth[valid]
                    lo = np.quantile(vals, loq) if loq > 0.0 else 0.0
                    hi = np.quantile(vals, hiq) if hiq < 1.0 else np.max(vals)
                    if self.crop_mode == 'mask':
                        mask_low = depth < lo
                        mask_high = depth > hi
                        depth[mask_low | mask_high] = 0.0
                    else:
                        # default: clip to [lo, hi]
                        depth = np.clip(depth, a_min=lo, a_max=hi)
                except Exception:
                    # ignore cropping on failure
                    pass

        # Room-scale normalization: map chosen quantile to target depth
        if self.enable_room_scale:
            valid = depth > 0.0
            if np.any(valid):
                try:
                    q = np.clip(self.room_depth_q, 0.0, 1.0)
                    target = max(1e-6, float(self.room_max_depth_m))
                    current = float(np.quantile(depth[valid], q))
                    if current > 1e-6:
                        scale = target / current
                        depth = depth * scale
                except Exception:
                    pass

        # final sanitization
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
        return depth

def box_depth_stats(depth: np.ndarray, bbox, ksize: int = 5):
    import numpy as np
    x1,y1,x2,y2 = map(int, bbox)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(depth.shape[1]-1, x2); y2 = min(depth.shape[0]-1, y2)
    if x2<=x1 or y2<=y1:
        return 0.0, 0.0
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    med = float(np.median(patch))
    var = float(np.var(patch))
    return med, var

def project_bbox_to_3d_centroid(bbox, depth: np.ndarray, cam) -> tuple:
    import numpy as np
    x1,y1,x2,y2 = bbox
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    z = float(np.median(depth[int(max(0,y1)):int(min(depth.shape[0],y2)), int(max(0,x1)):int(min(depth.shape[1],x2))]))
    if z <= 0.0:
        z = 1e-3
    X = (u - cam.cx) * z / cam.fx
    Y = (v - cam.cy) * z / cam.fy
    return (float(X), float(Y), float(z))
