from typing import List, Dict, Any, Optional

class OpenVocabDetector:
    def __init__(
        self,
        backend: str = "auto",
        yolo_weights: Optional[str] = None,
        owlvit_model: str = "google/owlvit-base-patch32",
        device: str = "cuda",
        groundingdino_config: Optional[str] = "thirdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        groundingdino_weights: Optional[str] = "thirdparty/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        confidence_threshold: float = 0.3,
        # groundingdino_config: Optional[str] = "thirdparty/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        # groundingdino_weights: Optional[str] = "thirdparty/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
    ):
        """Unified open-vocabulary detector wrapper.

        Parameters:
            backend: one of {"auto","owlvit","groundingdino","yolo"}
            yolo_weights: path or model name for YOLO (ultralytics)
            owlvit_model: HF model id for OwlViT
            device: torch device string
            groundingdino_config: path to GroundingDINO config (.py)
            groundingdino_weights: path to GroundingDINO checkpoint (.pth / .pt)
        Raises:
            RuntimeError if a designated backend cannot be initialized.
        """
        self.backend = backend
        self.yolo_weights = yolo_weights
        self.owlvit_model = owlvit_model
        self.device = device
        self.groundingdino_config = groundingdino_config
        self.groundingdino_weights = groundingdino_weights
        # Minimum score to keep a detection (0-1). If detector does not provide scores,
        # we keep returned items as-is.
        self.confidence_threshold = float(confidence_threshold)
        self._init_backends()

    def _init_backends(self):
        self._owl_processor = None
        self._owl_model = None
        self._yolo = None
        self._dino = None

        errors: Dict[str, str] = {}

        # Attempt OwlViT
        if self.backend in ("auto", "owlvit"):
            try:
                from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore
                self._owl_processor = OwlViTProcessor.from_pretrained(self.owlvit_model)
                self._owl_model = OwlViTForObjectDetection.from_pretrained(self.owlvit_model).to(self.device)
                if self.backend == "auto":
                    self.backend = "owlvit"
            except Exception as e:  # pragma: no cover
                errors["owlvit"] = f"OwlViT init failed: {e}"  # record but only raise if explicitly requested
                if self.backend == "owlvit":
                    raise RuntimeError(errors["owlvit"]) from e

        # Attempt GroundingDINO
        if self.backend in ("auto", "groundingdino"):
            try:
                import groundingdino  # type: ignore
                from groundingdino.util.inference import Model as GDModel  # type: ignore
                if self.backend == "groundingdino":
                    if not self.groundingdino_config or not self.groundingdino_weights:
                        raise RuntimeError("GroundingDINO requires groundingdino_config and groundingdino_weights.")
                # For auto, require both to promote deterministic behavior
                if self.backend == "auto" and (self.groundingdino_config and self.groundingdino_weights):
                    self._gd_model = GDModel(
                        model_config_path=self.groundingdino_config,
                        model_checkpoint_path=self.groundingdino_weights,
                        device=self.device,
                    )
                    self._dino = groundingdino
                    self.backend = "groundingdino"
                elif self.backend == "groundingdino":
                    self._gd_model = GDModel(
                        model_config_path=self.groundingdino_config,
                        model_checkpoint_path=self.groundingdino_weights,
                        device=self.device,
                    )
                    self._dino = groundingdino
            except Exception as e:  # pragma: no cover
                errors["groundingdino"] = f"GroundingDINO init failed: {e}"
                if self.backend == "groundingdino":
                    raise RuntimeError(errors["groundingdino"]) from e

        # Attempt YOLO
        if self.backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO  # type: ignore
                self._yolo = YOLO(self.yolo_weights or "yolov8n.pt")
                if self.backend == "auto":
                    self.backend = "yolo"
            except Exception as e:  # pragma: no cover
                errors["yolo"] = f"YOLO init failed: {e}"
                if self.backend == "yolo":
                    raise RuntimeError(errors["yolo"]) from e

        if self.backend == "auto":
            # None succeeded decisively
            detail = ", ".join(f"{k}: {v}" for k, v in errors.items()) or "no backends attempted"
            raise RuntimeError(f"Auto backend selection failed. Details: {detail}")

    def detect(self, frame, text_queries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.backend == "owlvit":
            if self._owl_model is None:
                raise RuntimeError("OwlViT backend not initialized.")
            return self._detect_owlvit(frame, text_queries or ["object", "tool", "cup", "bottle", "drawer", "door", "microwave", "robot", "arm", "gripper"])
        if self.backend == "groundingdino":
            if not hasattr(self, "_gd_model") or self._gd_model is None:
                raise RuntimeError("GroundingDINO backend not initialized.")
            return self._detect_groundingdino(frame, text_queries or ["object .", "robot arm .", "gripper .", "cup .", "cube .", "drawer .", "microwave ."])
        if self.backend == "yolo":
            if self._yolo is None:
                raise RuntimeError("YOLO backend not initialized.")
            return self._detect_yolo(frame)
        raise RuntimeError(f"Unsupported backend '{self.backend}'.")

    def _detect_owlvit(self, frame, queries: List[str]):
        import torch
        from transformers import OwlViTProcessor
        import numpy as np, cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        img = Image.fromarray(rgb)
        text = [[q] for q in queries]
        inputs = self._owl_processor(text=text, images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._owl_model(**inputs)
        results = self._owl_processor.post_process_object_detection(outputs=outputs, target_sizes=[img.size[::-1]], threshold=0.1)[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        out = []
        for b,s,l in zip(boxes, scores, labels):
            name = queries[int(l)][0] if isinstance(queries[0], list) else queries[int(l)]
            sc = float(s)
            if sc < self.confidence_threshold:
                continue
            out.append({"name": name, "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "score": sc})
        return out

    def _detect_groundingdino(self, frame, queries: List[str]):
        if not hasattr(self, "_gd_model") or self._gd_model is None:
            raise RuntimeError("GroundingDINO model not initialized.")
        caption = ". ".join(queries)
        detections, phrases = self._gd_model.predict_with_caption(
            image=frame,
            caption=caption,
            box_threshold=0.35,
            text_threshold=0.25,
        )
        boxes = getattr(detections, 'xyxy', None)
        confidences = getattr(detections, 'confidence', None)
        if boxes is None:
            raise RuntimeError("GroundingDINO returned no boxes attribute.")
        out: List[Dict[str, Any]] = []
        for i, xy in enumerate(boxes):
            x0, y0, x1, y1 = [float(v) for v in xy]
            score = float(confidences[i]) if confidences is not None and len(confidences) > i else 0.0
            name = phrases[i] if i < len(phrases) else 'object'
            if score >= self.confidence_threshold:
                out.append({'name': name, 'bbox': [x0, y0, x1, y1], 'score': score})
        return sorted(out, key=lambda d: d['score'], reverse=True)[:20]

    def _detect_yolo(self, frame):
        res = self._yolo.predict(frame, imgsz=640, verbose=False)
        boxes = []
        for r in res:
            if r.boxes is None: 
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                name = r.names.get(cls_id, f'cls{cls_id}')
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf.item())
                if conf >= self.confidence_threshold:
                    boxes.append({'name': name, 'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])], 'score': conf})
        boxes = sorted(boxes, key=lambda d: d['score'], reverse=True)[:20]
        return boxes

