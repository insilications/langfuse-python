from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import rich
from rich.pretty import pprint

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from langchain_core.messages import BaseMessage

    langchain_normalization: bool = True

    def normalize_message_to_dict(message: BaseMessage) -> dict[str, Any]:
        """
        Converts a LangChain `BaseMessage` into a standardized dictionary representation

        Unlike a standard Pydantic `model_dump()`,
        this function deliberately intercepts and reconstructs the fields:
        - `content`: explicitly populated using `message.content_blocks`
        rather than `message.content`. The `content` attribute passes through the raw,
        provider-native format.
        The new `content_blocks` attribute provides a standardized representation
        of all message content types, regardless of provider (OpenAI, Anthropic, etc.).
        It is fully compatible with existing LangChain applications.
        https://docs.langchain.com/oss/python/langchain/messages#standard-content-blocks
        - `name`: extracted from `additional_kwargs` and promoted to a top-level key

        Args:
            message (BaseMessage): The LangChain message instance to convert.

        Returns:
            dict[str, Any]: A JSON-compatible dictionary representation of the message.
        """

        rich.print("\n--- START normalize_message_to_dict ---\n")
        rich.print(f"f\n===== {type(message).__name__} ===== EITA")

        rich.print("message:\n")
        pprint(
            message,
            expand_all=True,
            indent_guides=False,
            max_string=2000,
        )
        rich.print("=====")

        message_dict: dict[str, Any] = message.model_dump(
            mode="json", exclude={"content"}
        )
        message_dict["content"] = message.content_blocks

        additional_kwargs = message.additional_kwargs
        if name := additional_kwargs.get("name"):
            message_dict["name"] = name

        rich.print("message_dict:\n")
        pprint(
            message_dict,
            expand_all=True,
            indent_guides=False,
            max_string=2000,
        )
        rich.print("==========================")
        rich.print("\n--- END normalize_message_to_dict ---\n")
        return message_dict

    def normalize_message_list_to_dicts(
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        return [normalize_message_to_dict(m) for m in messages]

    def normalize_nested_messages(
        io: Any,
    ) -> Any:
        """
        Recursively traverses arbitrary data structures
        to find and convert embedded LangChain `BaseMessage` objects into dictionaries.

        Clones the traversed containers to guarantee
        that the original input is never mutated.
        It enforces a maximum recursion depth of 10 levels
        to protect against stack overflows from cyclical references.

        - Dataclasses are parsed field-by-field. The built-in `dataclasses.asdict()`
        is intentionally avoided to prevent performance penalties
        from redundant double-traversals and deep copies.
        - Dictionary keys are preserved as-is; only dictionary values are traversed.

        Args:
            io (Any): The input payload.

        Returns:
            Any: A new data structure identical to the input, but with all nested
                `BaseMessage` instances replaced by their normalized dictionary formats.
                If max depth is exceeded or the type is primitive,
                the value is returned as-is.
        """

        convert_msg_to_dict: Callable[[BaseMessage], dict[str, Any]] = (
            normalize_message_to_dict
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
                return {k: walk(v, level + 1) for k, v in value.items()}

            if isinstance(value, list):
                return [walk(v, level + 1) for v in value]

            return value

        return walk(io, 0)

except ImportError:
    langchain_normalization: bool = False

    # Fallback: Langchain isn't available, so we define a no-op conversion function
    def normalize_nested_messages(io: Any) -> Any:
        return io
