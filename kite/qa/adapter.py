import asyncio
import base64
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import httpx
from httpx import Timeout
import os

class RoboFACAdapter:
    def __init__(self):
        self.SEMAPHORE_SIZE = 30
        self.timeout = Timeout(connect=10.0, read=200.0, write=30.0, pool=10.0)

    def _frames_to_content(self, frames: List[np.ndarray], prompt: str) -> List[Dict[str,Any]]:
        content = []
        for fr in frames:
            fr512 = cv2.resize(fr, (512,512))
            ok, buf = cv2.imencode('.jpg', fr512)
            b64 = base64.b64encode(buf).decode('utf-8')
            content.append({"type":"image_url","image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail":"low"}})
        content.append({"type":"text","text": prompt})
        return content

    def _read_window_frames(self, video_path: str, t0: float, t1: float, fps_hint: Optional[float]=None, step: int=4, max_frames:int=12) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = int(t0*fps); end_f = int(t1*fps)
        frames = []; idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        while True:
            ok, fr = cap.read()
            if not ok: break
            fno = start_f + idx
            if fno>=end_f: break
            if (idx % step)==0:
                frames.append(fr.copy())
                if len(frames)>=max_frames: break
            idx += 1
        cap.release()
        return frames

    async def _async_api_request(self, client, prompt_content, model_name: str, url: str, semaphore: asyncio.Semaphore):
        """We only need to send a single chat completion request where the "content" is
        already prepared (list of image/text dicts). Returns the text result.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            # Request a longer, more deterministic completion to follow the prompt requirements
            # "max_tokens": 512,
            # "temperature": 0.1,
            # "top_p": 1.0,
            # "n": 1,
            "response_format": {"type": "text"}
        }
        max_retries = int(os.environ.get('KITE_API_MAX_RETRIES', '8'))
        attempt = 0
        last_err = None
        # Retry with exponential backoff (cap) so UI can surface failures instead of hanging silently
        while attempt < max_retries:
            try:
                async with semaphore:
                    response = await client.post(f"{url}/chat/completions", headers=headers, json=payload, timeout=10000)
                    response_json = response.json()
                    return response_json["choices"][0]["message"]["content"]
            except Exception as e:
                attempt += 1
                last_err = e
                await asyncio.sleep(min(4, 0.5 * (2 ** (attempt-1))))
        raise RuntimeError(f"Model API request failed after {max_retries} retries to {url}: {last_err}")

    async def _call_model(self, model_name: str, model_url: str, prompt_content: List[Dict[str,Any]]):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            sem = asyncio.Semaphore(self.SEMAPHORE_SIZE)
            res = await self._async_api_request(client, prompt_content, model_name=model_name, url=model_url, semaphore=sem)
            return res


    # async def locate(self, model_name: str, model_url: str, video_path: str, locate_prompt: str, t0: float, t1: float, fps_hint: Optional[float]=None) -> Dict[str,Any]:
    #     frames = self._read_window_frames(video_path, t0, t1, fps_hint=fps_hint, step=2, max_frames=12)
    #     prompt_content = self._frames_to_content(frames, locate_prompt)
    #     text = await self._call_model(model_name, model_url, prompt_content)
    #     return {"raw": text}

    # async def identify(self, model_name: str, model_url: str, video_path: str, identify_prompt: str, t0: float, t1: float, fps_hint: Optional[float]=None) -> Dict[str,Any]:
    #     frames = self._read_window_frames(video_path, t0, t1, fps_hint=fps_hint, step=2, max_frames=12)
    #     prompt_content = self._frames_to_content(frames, identify_prompt)
    #     text = await self._call_model(model_name, model_url, prompt_content)
    #     return {"raw": text}

    # async def freeform_with_images(self, model_name: str, model_url: str, images: List[np.ndarray], prompt_text: str) -> str:
    #     prompt_content = self._frames_to_content(images, prompt_text)
    #     return await self._call_model(model_name, model_url, prompt_content)

    async def qa_with_images_and_context(self, model_name: str, model_url: str, frames: List[np.ndarray], question_text: str, context_text: str) -> str:
        prompt = f"{question_text}\n\n[CONTEXT]\n{context_text}" if context_text else question_text
        prompt_content = self._frames_to_content(frames, prompt)
        return await self._call_model(model_name, model_url, prompt_content)



    def make_eval_prompt(self, question, pred, ref):
        return f"""You are an expert evaluator. Assess the quality of a model's response to the user's query.

            Question: {question}

            Reference answer: {ref}

            Model's response: {pred}

            Evaluate the model's response on the following criteria: 
            - correctness: factual accuracy and consistency with the reference answer.
            - relevance: how well the model's response addresses the question.
            - completeness: whether all key aspects of the reference answer are covered.
        
            For each criterion, provide a score from 0 to 5 and a **brief** explanation, the score should be an integer.
            The score you give needs to be strict and demanding.

            Output ONLY the JSON object in the following format:
            {{
            "criteria": {{
                "correctness": {{"score": <0-5>, "explanation": <brief explanation>}},
                "relevance": {{"score": <0-5>, "explanation": <brief explanation>}},
                "completeness": {{"score": <0-5>, "explanation": <brief explanation>}},
            }}
            }}
            """