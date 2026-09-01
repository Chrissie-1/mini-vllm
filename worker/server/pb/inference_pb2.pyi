from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateRequest(_message.Message):
    __slots__ = ("prompt", "max_tokens", "temperature", "top_k", "top_p", "request_id")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    request_id: str
    def __init__(self, prompt: _Optional[str] = ..., max_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ..., request_id: _Optional[str] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("text", "prompt_tokens", "completion_tokens", "finish_reason", "request_id")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    request_id: str
    def __init__(self, text: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., finish_reason: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ("text", "token_id", "done", "finish_reason")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    text: str
    token_id: int
    done: bool
    finish_reason: str
    def __init__(self, text: _Optional[str] = ..., token_id: _Optional[int] = ..., done: _Optional[bool] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "model", "running", "queued", "max_batch_size")
    READY_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RUNNING_FIELD_NUMBER: _ClassVar[int]
    QUEUED_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    model: str
    running: int
    queued: int
    max_batch_size: int
    def __init__(self, ready: _Optional[bool] = ..., model: _Optional[str] = ..., running: _Optional[int] = ..., queued: _Optional[int] = ..., max_batch_size: _Optional[int] = ...) -> None: ...
