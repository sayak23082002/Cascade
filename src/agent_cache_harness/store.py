import hashlib
import json
from pathlib import Path
from typing import Optional

class LocalDiskStore:
    def __init__(self, cache_dir: str = ".agent_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_key(self, model_name: str, payload: dict) -> str:
        # sort_keys=True is critical to prevent cache misses from identical dicts with different orders
        serialized = json.dumps(
            {"model": model_name, "payload": payload}, 
            sort_keys=True, 
            separators=(',', ':')
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, model_name: str, payload: dict) -> Optional[str]:
        key = self._generate_key(model_name, payload)
        file_path = self.cache_dir / f"{key}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f).get("response")
        return None

    def set(self, model_name: str, payload: dict, response: str) -> None:
        key = self._generate_key(model_name, payload)
        with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump({"response": response}, f)