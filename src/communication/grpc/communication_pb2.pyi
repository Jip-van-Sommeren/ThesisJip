from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INFORM: _ClassVar[MessageType]
    REQUEST: _ClassVar[MessageType]
    REPLY: _ClassVar[MessageType]
    BROADCAST: _ClassVar[MessageType]
    ERROR: _ClassVar[MessageType]
INFORM: MessageType
REQUEST: MessageType
REPLY: MessageType
BROADCAST: MessageType
ERROR: MessageType

class Message(_message.Message):
    __slots__ = ("message_id", "sender_id", "receiver_id", "message_type", "content", "timestamp", "reply_to")
    class ContentEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    REPLY_TO_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: _containers.ScalarMap[str, str]
    timestamp: float
    reply_to: str
    def __init__(self, message_id: _Optional[str] = ..., sender_id: _Optional[str] = ..., receiver_id: _Optional[str] = ..., message_type: _Optional[_Union[MessageType, str]] = ..., content: _Optional[_Mapping[str, str]] = ..., timestamp: _Optional[float] = ..., reply_to: _Optional[str] = ...) -> None: ...

class SendMessageRequest(_message.Message):
    __slots__ = ("sender_id", "receiver_id", "message_type", "content", "reply_to")
    class ContentEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    REPLY_TO_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: _containers.ScalarMap[str, str]
    reply_to: str
    def __init__(self, sender_id: _Optional[str] = ..., receiver_id: _Optional[str] = ..., message_type: _Optional[_Union[MessageType, str]] = ..., content: _Optional[_Mapping[str, str]] = ..., reply_to: _Optional[str] = ...) -> None: ...

class SendMessageResponse(_message.Message):
    __slots__ = ("success", "message_id", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message_id: str
    error_message: str
    def __init__(self, success: bool = ..., message_id: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class BroadcastMessageRequest(_message.Message):
    __slots__ = ("sender_id", "message_type", "content")
    class ContentEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    message_type: MessageType
    content: _containers.ScalarMap[str, str]
    def __init__(self, sender_id: _Optional[str] = ..., message_type: _Optional[_Union[MessageType, str]] = ..., content: _Optional[_Mapping[str, str]] = ...) -> None: ...

class BroadcastMessageResponse(_message.Message):
    __slots__ = ("success", "delivered", "failed", "total_targets", "message_ids")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    DELIVERED_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TARGETS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    delivered: int
    failed: int
    total_targets: int
    message_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, success: bool = ..., delivered: _Optional[int] = ..., failed: _Optional[int] = ..., total_targets: _Optional[int] = ..., message_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class GetMessagesRequest(_message.Message):
    __slots__ = ("agent_id", "clear_mailbox")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    CLEAR_MAILBOX_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    clear_mailbox: bool
    def __init__(self, agent_id: _Optional[str] = ..., clear_mailbox: bool = ...) -> None: ...

class GetMessagesResponse(_message.Message):
    __slots__ = ("messages", "count")
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    count: int
    def __init__(self, messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., count: _Optional[int] = ...) -> None: ...

class RegisterAgentRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    def __init__(self, agent_id: _Optional[str] = ...) -> None: ...

class RegisterAgentResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class AddLinkRequest(_message.Message):
    __slots__ = ("sender_id", "receiver_id")
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    receiver_id: str
    def __init__(self, sender_id: _Optional[str] = ..., receiver_id: _Optional[str] = ...) -> None: ...

class AddLinkResponse(_message.Message):
    __slots__ = ("success", "error_message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_message: str
    def __init__(self, success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class StatisticsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatisticsResponse(_message.Message):
    __slots__ = ("messages_sent", "messages_delivered", "delivery_failures", "avg_delivery_time")
    MESSAGES_SENT_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_DELIVERED_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_FAILURES_FIELD_NUMBER: _ClassVar[int]
    AVG_DELIVERY_TIME_FIELD_NUMBER: _ClassVar[int]
    messages_sent: int
    messages_delivered: int
    delivery_failures: int
    avg_delivery_time: float
    def __init__(self, messages_sent: _Optional[int] = ..., messages_delivered: _Optional[int] = ..., delivery_failures: _Optional[int] = ..., avg_delivery_time: _Optional[float] = ...) -> None: ...

class TopologyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CommunicationLink(_message.Message):
    __slots__ = ("sender_id", "receiver_id")
    SENDER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    sender_id: str
    receiver_id: str
    def __init__(self, sender_id: _Optional[str] = ..., receiver_id: _Optional[str] = ...) -> None: ...

class TopologyResponse(_message.Message):
    __slots__ = ("links", "total_links")
    LINKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LINKS_FIELD_NUMBER: _ClassVar[int]
    links: _containers.RepeatedCompositeFieldContainer[CommunicationLink]
    total_links: int
    def __init__(self, links: _Optional[_Iterable[_Union[CommunicationLink, _Mapping]]] = ..., total_links: _Optional[int] = ...) -> None: ...
