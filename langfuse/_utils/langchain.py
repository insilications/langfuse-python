from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from langchain_core.messages import BaseMessage

    def convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
        message_dict: dict[str, Any] = message.model_dump(
            mode="json", exclude={"content"}
        )
        message_dict["content"] = message.content_blocks

        if "name" in message.additional_kwargs:  # type: ignore
            message_dict["name"] = message.additional_kwargs["name"]  # type: ignore

        if message.additional_kwargs:  # type: ignore
            message_dict["additional_kwargs"] = message.additional_kwargs  # type: ignore

        return message_dict

    def convert_langchain_io_to_dict(
        io: Any | None,
    ) -> Any | None:
        convert_msg_to_dict: Callable[[BaseMessage], dict[str, Any]] = (
            convert_message_to_dict
        )
        max_levels = 10

        def walk(
            value: Any,
            level: int,
        ) -> Any:
            if level > max_levels:
                return value

            # 1. `BaseMessage` to `Dict[str, Any]` before handling `dataclass`
            if isinstance(value, BaseMessage):
                return convert_msg_to_dict(value)

            # 2. `dataclass` instances field-by-field to `dict[str, Any]`
            # Avoid using `asdict()` because it performs double traversal/deepcopy
            if is_dataclass(value) and not isinstance(value, type):
                return {
                    f.name: walk(getattr(value, f.name), level + 1)
                    for f in fields(value)
                }

            # 3. Containers -> always produce new containers (no mutation)
            if isinstance(value, dict):
                # Keys intentionally not walked
                return {k: walk(v, level + 1) for k, v in value.items()}  # type: ignore

            if isinstance(value, list):
                return [walk(v, level + 1) for v in value]  # type: ignore

            return value

        if io is None:
            return io
        return walk(io, 0)

except ImportError:
    # Fallback: Langchain isn't available, so we define a no-op conversion function
    def convert_langchain_io_to_dict(io: Any | None) -> Any | None:
        return io
