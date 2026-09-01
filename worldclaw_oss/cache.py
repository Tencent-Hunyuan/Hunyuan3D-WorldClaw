from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_key(prompt: str, reference_sha256: str, model_id: str, revision: str, parameters: dict[str, Any]) -> str:
    payload={"model_id":model_id,"parameters":parameters,"prompt":prompt,"reference_sha256":reference_sha256,"revision":revision}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


class ContentCache:
    def __init__(self, root: Path): self.root=Path(root)
    def path(self,key: str,suffix: str) -> Path: return self.root/key[:2]/f"{key}.{suffix.lstrip('.')}"
    def put_bytes(self,key: str,suffix: str,data: bytes) -> Path:
        path=self.path(key,suffix); path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}"); temp.write_bytes(data); os.replace(temp,path); return path
    def get(self,key: str,suffix: str) -> Path | None:
        path=self.path(key,suffix); return path if path.is_file() else None
