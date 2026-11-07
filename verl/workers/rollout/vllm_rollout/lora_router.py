"""Shared per-batch LoRA routing for vLLM rollout backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


TEACHER_LORA_META_KEY = "cpsd_teacher_lora_requested"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


@dataclass(frozen=True)
class TeacherLoRAConfig:
    path: Optional[str]
    rank: int
    name: str
    int_id: int

    @classmethod
    def from_rollout_config(cls, config: Any) -> "TeacherLoRAConfig":
        raw_path = _config_get(config, "teacher_lora_path", None)
        path = str(raw_path).strip() if raw_path else None
        return cls(
            path=path,
            rank=int(_config_get(config, "teacher_lora_rank", 0) or 0),
            name=str(_config_get(config, "teacher_lora_name", "cpsd_teacher") or "").strip(),
            int_id=int(_config_get(config, "teacher_lora_int_id", 1001) or 0),
        )

    def validate(self) -> None:
        if not self.path:
            return
        if self.rank <= 0:
            raise ValueError("teacher_lora_rank must be positive when teacher_lora_path is set")
        if not self.name:
            raise ValueError("teacher_lora_name must be non-empty when teacher_lora_path is set")
        if self.int_id <= 0:
            raise ValueError("teacher_lora_int_id must be positive when teacher_lora_path is set")


def resolve_lora_engine_kwargs(
    rollout_config: Any,
    *,
    actor_lora_enabled: bool,
    actor_lora_rank: int,
) -> dict[str, Any]:
    """Return vLLM engine LoRA kwargs and reject unsupported dual adapters."""
    teacher = TeacherLoRAConfig.from_rollout_config(rollout_config)
    teacher.validate()
    if actor_lora_enabled and teacher.path:
        raise ValueError(
            "actor LoRA and external teacher LoRA cannot be enabled together"
        )
    if actor_lora_enabled:
        return {
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": int(actor_lora_rank),
        }
    if teacher.path:
        return {
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": teacher.rank,
        }
    return {}


class LoRARouter:
    """Route legacy actor LoRA globally or an external adapter to teacher rows."""

    def __init__(
        self,
        rollout_config: Any,
        *,
        actor_lora_enabled: bool,
        request_factory: Callable[..., Any],
    ) -> None:
        self.actor_lora_enabled = bool(actor_lora_enabled)
        self.request_factory = request_factory
        self.teacher = TeacherLoRAConfig.from_rollout_config(rollout_config)
        self.teacher.validate()
        if self.actor_lora_enabled and self.teacher.path:
            raise ValueError(
                "actor LoRA and external teacher LoRA cannot be enabled together"
            )

    @staticmethod
    def _teacher_flags(meta_info: Mapping[str, Any], batch_size: int) -> list[bool]:
        raw = meta_info.get(TEACHER_LORA_META_KEY, False)
        if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes)):
            raw = raw.tolist()
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != batch_size:
                raise ValueError(
                    f"{TEACHER_LORA_META_KEY} has {len(raw)} rows for batch size {batch_size}"
                )
            return [bool(value) for value in raw]
        return [bool(raw)] * batch_size

    def requests(
        self,
        meta_info: Mapping[str, Any],
        batch_size: int,
        *,
        loaded_actor_lora_ids: Sequence[int] = (),
    ) -> Optional[list[Any]]:
        if batch_size <= 0:
            return None
        if self.teacher.path:
            flags = self._teacher_flags(meta_info, batch_size)
            if not any(flags):
                return None
            request = self.request_factory(
                lora_name=self.teacher.name,
                lora_int_id=self.teacher.int_id,
                lora_path=self.teacher.path,
            )
            return [request if requested else None for requested in flags]
        if not self.actor_lora_enabled:
            return None
        loaded = list(loaded_actor_lora_ids)
        if not loaded:
            return None
        lora_int_id = int(loaded[0])
        request = self.request_factory(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="/simon-stub-path",
        )
        return [request] * batch_size
