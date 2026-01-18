"""
MAS Organization Layer

Organizational structure components based on formal definitions:
- Role: R = (responsibilities, permissions, requirements, expectations)
- Group: G = (Name, Roles_G, Purpose)
- Hierarchy: Sup: A → A ∪ {⊥} and Sub: A → 2^A
"""

from .role import (
    Role,
    RoleManager,
    Responsibility,
    Permission,
    PermissionType,
    Requirement,
    Expectation,
    ExpectationType,
)
from .group import Group, GroupManager
from .hierarchy import (
    HierarchyManager,
    OrganizationalPosition,
    HierarchyMessage,
    MessageType,
)

__all__ = [
    # Role system
    "Role",
    "RoleManager",
    "Responsibility",
    "Permission",
    "PermissionType",
    "Requirement",
    "Expectation",
    "ExpectationType",
    # Group system
    "Group",
    "GroupManager",
    # Hierarchy system
    "HierarchyManager",
    "OrganizationalPosition",
    "HierarchyMessage",
    "MessageType",
]
