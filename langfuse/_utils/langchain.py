from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast, overload

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

    @overload
    def convert_langchain_io_to_dict(io: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def convert_langchain_io_to_dict[T](
        io: list[T],
    ) -> list[dict[str, Any]]: ...

    @overload
    def convert_langchain_io_to_dict(
        io: None,
    ) -> None: ...

    def convert_langchain_io_to_dict[T](  # type: ignore
        io: dict[str, Any] | list[T] | None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if io is None:
            return io

        convert_msg_to_dict: Callable[[Any], dict[str, Any]] = convert_message_to_dict
        max_levels = 10

        def walk[U](
            value: dict[str, Any] | list[U] | U,
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
                # Pylance lost the type parameters after isinstance. We restore them.
                if TYPE_CHECKING:
                    value = cast("dict[str, Any]", value)
                return {k: walk(v, level + 1) for k, v in value.items()}

            if isinstance(value, list):
                # Restore the list[U] generic type parameter
                if TYPE_CHECKING:
                    value = cast("list[U]", value)
                return [walk(v, level + 1) for v in value]

            return value

        return walk(io, 0)

    # def convert_langchain_io_to_dict(
    #     io: Any | None,
    # ) -> Any | None:
    #     convert_msg_to_dict: Callable[[BaseMessage], dict[str, Any]] = (
    #         convert_message_to_dict
    #     )
    #     max_levels = 10

    #     def walk(
    #         value: Any,
    #         level: int,
    #     ) -> Any:
    #         if level > max_levels:
    #             return value

    #         # 1. `BaseMessage` to `Dict[str, Any]` before handling `dataclass`
    #         if isinstance(value, BaseMessage):
    #             return convert_msg_to_dict(value)

    #         # 2. `dataclass` instances field-by-field to `dict[str, Any]`
    #         # Avoid using `asdict()` because it performs double traversal/deepcopy
    #         if is_dataclass(value) and not isinstance(value, type):
    #             return {
    #                 f.name: walk(getattr(value, f.name), level + 1)
    #                 for f in fields(value)
    #             }

    #         # 3. Containers -> always produce new containers (no mutation)
    #         if isinstance(value, dict):
    #             # Keys intentionally not walked
    #             return {k: walk(v, level + 1) for k, v in value.items()}  # type: ignore

    #         if isinstance(value, list):
    #             return [walk(v, level + 1) for v in value]  # type: ignore

    #         return value

    #     if io is None:
    #         return io
    #     return walk(io, 0)

except ImportError:
    # Fallback: Langchain isn't available, so we define a no-op conversion function
    @overload
    def convert_langchain_io_to_dict(io: dict[str, Any]) -> dict[str, Any]: ...

    @overload
    def convert_langchain_io_to_dict[T](
        io: list[T],
    ) -> list[dict[str, Any]]: ...

    @overload
    def convert_langchain_io_to_dict(
        io: None,
    ) -> None: ...

    def convert_langchain_io_to_dict[T](
        io: dict[str, Any] | list[T] | None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        return cast("dict[str, Any] | list[dict[str, Any]] | None", io)

    # def convert_langchain_io_to_dict(io: Any | None) -> Any | None:
    #     return io
