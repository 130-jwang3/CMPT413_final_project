# BigModel/Ollama/Ollama.py  (full drop-in)
import os
import base64
import mimetypes
import requests

from BigModel.Base import BigModelBase

def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:image")

def _is_file(s: str) -> bool:
    return isinstance(s, str) and os.path.isfile(s)

def _b64_from_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

class Chat_Ollama(BigModelBase):
    """
    Minimal chat adapter for Ollama.
    - Works with text-only and vision models.
    - Supports image inputs as:
        * data URLs (data:image/...;base64,<...>)
        * local file paths (we base64-encode)
        * http(s) URLs (we fetch and base64-encode)
    - Omits 'images' key entirely for text-only messages.
    - Uses format:'json' for text models, but disables it for llava/vision.
    """
    def __init__(self, host=None, model="llava:latest", max_tokens=4096, temperature=0.1, top_p=0.3):
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = host.rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", model)

        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.tokens_per_minute = int(os.environ.get("OLLAMA_TPM", "1000000"))
        self.used_token = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        self.last_used_token = {"total_tokens": 0}

        self.image_rotation = 0
        self.image_rotation_angle = 0
        self.http_timeout = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "600"))

        # Use format:'json' by default, but disable automatically for common vision models
        name = (self.model or "").lower()
        self.use_json_format = not any(tag in name for tag in ("llava", "vision", "vl"))

    def set_image_rotation(self, angle: int):
        self.image_rotation = angle
        self.image_rotation_angle = angle

    def _mime_from_path(self, path: str) -> str:
        mt, _ = mimetypes.guess_type(path)
        return mt or "image/png"

    def _b64_from_file(self, path: str) -> str:
        try:
            return self.image_encoder_base64(path)
        except Exception:
            with open(path, "rb") as f:
                return _b64_from_bytes(f.read())

    def _b64_from_url(self, url: str) -> str:
        r = requests.get(url, timeout=self.http_timeout)
        r.raise_for_status()
        return _b64_from_bytes(r.content)

    def _b64_from_data_url(self, data_url: str) -> str:
        return data_url.split("base64,", 1)[-1]

    def _coerce_image_to_b64(self, val: str) -> str | None:
        try:
            if _is_data_url(val):
                return self._b64_from_data_url(val)
            if _is_file(val):
                return self._b64_from_file(val)
            if _is_url(val):
                return self._b64_from_url(val)
            return None
        except Exception:
            return None

    def chat(self, message: list) -> str:
        conv = []
        for m in message:
            role = m.get("role", "user")
            texts, images_b64 = [], []

            for c in m.get("content", []):
                ctype = c.get("type")
                if ctype == "text":
                    texts.append(c.get("text", ""))

                elif ctype == "image_url":
                    url_field = c.get("image_url")
                    url = url_field.get("url", "") if isinstance(url_field, dict) else (url_field or "")
                    if url:
                        b64 = self._coerce_image_to_b64(url)
                        if b64:
                            images_b64.append(b64)

                elif ctype in ("image_path", "image"):
                    p = c.get("image_path") or c.get("url") or ""
                    if p:
                        b64 = self._coerce_image_to_b64(p)
                        if b64:
                            images_b64.append(b64)

            content = "\n\n".join([t for t in texts if t]).strip() or " "
            msg = {"role": role, "content": content}
            if images_b64:
                msg["images"] = images_b64
            conv.append(msg)

        schema = {
            "type": "object",
            "properties": {
                "normal_pattern": {"type": "string"},
                "abnormal_index": {"type": "string"},
                "abnormal_description": {"type": "string"},
                "abnormal_type_description": {"type": "string"}
            },
            "required": []  # allow either Task1 or Task2 shape
        }

        payload = {
            "model": self.model,
            "messages": conv,
            "stream": False,
            "format": schema,  # <— schema instead of "json"
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }

        if self.use_json_format:
            payload["format"] = "json"

        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.http_timeout)
        resp.raise_for_status()
        data = resp.json()
        out_txt = data.get("message", {}).get("content", "") or ""

        try:
            est_out = max(1, len(out_txt) // 4)
            self.used_token["completion_tokens"] += est_out
            self.used_token["total_tokens"] = self.used_token["completion_tokens"] + self.used_token["prompt_tokens"]
            self.last_used_token = {"total_tokens": est_out}
        except Exception:
            self.last_used_token = {"total_tokens": 0}

        return out_txt

    def text_item(self, content: str):
        return {"type": "text", "text": content}

    def image_item_from_path(self, image_path: str, detail: str = "high"):
        try:
            image_base64 = self._b64_from_file(image_path)
        except Exception:
            image_base64 = self.image_encoder_base64(image_path)
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}

    def image_item_from_base64(self, image_base64: str, detail: str = "high"):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}

    def get_used_token(self):
        return self.used_token

    def get_last_used_token(self):
        return self.last_used_token
