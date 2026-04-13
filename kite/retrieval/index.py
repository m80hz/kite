import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class SimpleHistIndexer:
    """CLIP-free fallback: HSV histogram embeddings per frame window."""
    def __init__(self, bins: int = 8):
        self.embs = []
        self.meta = []
        self.bins = bins

    def _embed_frame(self, frame):
        import cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[self.bins]*3,[0,180,0,256,0,256])
        hist = cv2.normalize(hist, None).flatten().astype('float32')
        return hist

    def add_window(self, frames: List[np.ndarray], label_letter: str, meta: Dict[str,Any]):
        # embed first and last frame
        vecs = []
        for fr in frames[:4] + frames[-4:]:
            vecs.append(self._embed_frame(fr))
        emb = np.mean(vecs, axis=0).astype('float32')
        self.embs.append(emb)
        self.meta.append({"label": label_letter, **meta})

    def build(self):
        self.embs = np.vstack(self.embs) if self.embs else np.zeros((0, self.bins**3), dtype='float32')

    def query(self, frames: List[np.ndarray], topk: int = 8) -> List[Dict[str,Any]]:
        if len(self.embs)==0:
            return []
        q = np.mean([self._embed_frame(fr) for fr in (frames[:4]+frames[-4:])], axis=0).astype('float32')
        # cosine similarity
        ef = (self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True)+1e-6))
        qf = q / (np.linalg.norm(q)+1e-6)
        sims = ef @ qf
        idxs = np.argsort(-sims)[:topk]
        return [{"idx": int(i), "score": float(sims[i]), "meta": self.meta[i]} for i in idxs]


# --- Lightweight retrieval utilities ---

def _embed_frames_hsv(frames: List[np.ndarray], bins: int = 8) -> np.ndarray:
    import cv2
    vecs = []
    for fr in (frames[:4] + frames[-4:]) if len(frames) > 0 else []:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[bins]*3,[0,180,0,256,0,256])
        hist = cv2.normalize(hist, None).flatten().astype('float32')
        vecs.append(hist)
    if not vecs:
        return np.zeros((bins**3,), dtype='float32')
    return np.mean(vecs, axis=0).astype('float32')


def _embed_frames_clip(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """Optional CLIP embedding (average over frames). Returns None if open_clip not available."""
    try:
        import torch
        import open_clip
        import cv2
        from PIL import Image
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()
        im_tensors = []
        for fr in (frames[:4] + frames[-4:]) if len(frames) > 0 else []:
            # BGR->RGB
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            im_tensors.append(preprocess(Image.fromarray(fr)))
        if not im_tensors:
            return None
        x = torch.stack(im_tensors)
        with torch.no_grad():
            feats = model.encode_image(x)
        emb = feats.mean(dim=0)
        emb = emb / (emb.norm() + 1e-6)
        return emb.cpu().numpy().astype('float32')
    except Exception:
        return None


def frames_similarity(frames_a: List[np.ndarray], frames_b: List[np.ndarray], backend: str = 'hsv') -> float:
    """Cosine similarity between two frame sets under chosen backend."""
    if backend == 'clip':
        ea = _embed_frames_clip(frames_a)
        eb = _embed_frames_clip(frames_b)
        if ea is not None and eb is not None:
            return float(np.dot(ea, eb) / (np.linalg.norm(ea)*np.linalg.norm(eb) + 1e-6))
        # fallback to hsv
    ea = _embed_frames_hsv(frames_a)
    eb = _embed_frames_hsv(frames_b)
    ea = ea / (np.linalg.norm(ea) + 1e-6)
    eb = eb / (np.linalg.norm(eb) + 1e-6)
    return float(ea @ eb)
