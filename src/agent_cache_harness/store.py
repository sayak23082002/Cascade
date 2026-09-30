import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

class BaseStore(ABC):
    def _generate_key(self, model_name: str, payload: Dict[str, Any]) -> str:
        """Centralized deterministic hashing for all storage backends."""
        serialized = json.dumps(
            {"model": model_name, "payload": payload}, 
            sort_keys=True, 
            separators=(',', ':')
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @abstractmethod
    def get(self, model_name: str, payload: Dict[str, Any]) -> Optional[str]:
        pass

    @abstractmethod
    def set(self, model_name: str, payload: Dict[str, Any], response: str) -> None:
        pass

class LocalDiskStore(BaseStore):
    def __init__(self, cache_dir: str = ".agent_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, model_name: str, payload: Dict[str, Any]) -> Optional[str]:
        key = self._generate_key(model_name, payload)
        file_path = self.cache_dir / f"{key}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f).get("response")
        return None

    def set(self, model_name: str, payload: Dict[str, Any], response: str) -> None:
        key = self._generate_key(model_name, payload)
        with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump({"response": response}, f)

class RedisStore(BaseStore):
    def __init__(self, redis_url: str, ttl_seconds: int = 604800, key_prefix: str = "agent_cache:"):
        """
        ttl_seconds: Default is 7 days.
        key_prefix: Isolates cache domains (e.g., "phase1:", "phase2:") within the same Redis instance.
        """
        import redis
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_seconds
        self.prefix = key_prefix

    def get(self, model_name: str, payload: Dict[str, Any]) -> Optional[str]:
        hash_key = self._generate_key(model_name, payload)
        redis_key = f"{self.prefix}{hash_key}"
        return self.client.get(redis_key)

    def set(self, model_name: str, payload: Dict[str, Any], response: str) -> None:
        hash_key = self._generate_key(model_name, payload)
        redis_key = f"{self.prefix}{hash_key}"
        self.client.setex(redis_key, self.ttl, response)