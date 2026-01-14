from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from simplecoder.permissions import (
    OperationType,
    PermissionLevel,
    PermissionManager,
    PermissionRequest,
)


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: JsonDict
    fn: Callable[..., Any]


def _workspace_root() -> Path:
    # Treat the current working directory as the workspace root.
    return Path(os.getcwd()).resolve()


def _resolve_path_in_workspace(path_str: str) -> Path:
    root = _workspace_root()
    p = (root / path_str).expanduser().resolve() if not os.path.isabs(path_str) else Path(path_str).expanduser().resolve()
    try:
        p.relative_to(root)
    except Exception as e:
        raise ValueError(f"Path escapes workspace root: {path_str}") from e
    return p


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def tool_list_files(directory: str = ".", pattern: str = "**/*", max_results: int = 200) -> JsonDict:
    """List files under a directory (workspace-scoped)."""
    base = _resolve_path_in_workspace(directory)
    if not base.exists():
        return {"directory": directory, "files": [], "error": "Directory does not exist"}
    if not base.is_dir():
        return {"directory": directory, "files": [], "error": "Not a directory"}

    files: list[str] = []
    # Path.rglob doesn't support "**/*" well if base is file; already handled.
    for p in base.rglob("*"):
        rel = str(p.relative_to(_workspace_root()))
        if fnmatch(rel, pattern) or fnmatch(p.name, pattern):
            files.append(rel)
        if len(files) >= max_results:
            break
    files.sort()
    return {"directory": str(base.relative_to(_workspace_root())), "files": files, "truncated": len(files) >= max_results}


def tool_read_file(filepath: str, start_line: int = 1, end_line: int = 200) -> JsonDict:
    """Read a file with optional line range."""
    path = _resolve_path_in_workspace(filepath)
    if not path.exists() or not path.is_file():
        return {"filepath": filepath, "error": "File not found"}

    lines = _read_text(path).splitlines()
    start = max(1, int(start_line))
    end = max(start, int(end_line))
    slice_lines = lines[start - 1 : end]
    return {
        "filepath": str(path.relative_to(_workspace_root())),
        "start_line": start,
        "end_line": min(end, len(lines)),
        "total_lines": len(lines),
        "content": "\n".join(slice_lines),
    }


def tool_search_files(
    query: str,
    directory: str = ".",
    pattern: str = "**/*.py",
    is_regex: bool = False,
    max_results: int = 20,
    context_lines: int = 2,
) -> JsonDict:
    """Search for text in files under a directory."""
    base = _resolve_path_in_workspace(directory)
    if not base.exists() or not base.is_dir():
        return {"matches": [], "error": "Directory not found"}

    rx = re.compile(query if is_regex else re.escape(query), re.IGNORECASE)
    matches: list[JsonDict] = []

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(_workspace_root()))
        if not fnmatch(rel, pattern) and not fnmatch(p.name, pattern):
            continue

        try:
            text = _read_text(p)
        except Exception:
            continue

        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if rx.search(line):
                start = max(1, idx - context_lines)
                end = min(len(lines), idx + context_lines)
                snippet = "\n".join(lines[start - 1 : end])
                matches.append(
                    {
                        "filepath": rel,
                        "line": idx,
                        "snippet": snippet,
                    }
                )
                if len(matches) >= max_results:
                    return {"query": query, "matches": matches, "truncated": True}

    return {"query": query, "matches": matches, "truncated": False}


def tool_write_file(filepath: str, content: str, overwrite: bool = False) -> JsonDict:
    """Create or overwrite a file."""
    path = _resolve_path_in_workspace(filepath)
    existed_before = path.exists()
    if path.exists() and not overwrite:
        return {"filepath": filepath, "error": "File exists (set overwrite=true)"}
    _write_text(path, content)
    return {
        "filepath": str(path.relative_to(_workspace_root())),
        "bytes": len(content.encode("utf-8")),
        "overwrote": existed_before,
    }


def tool_edit_file_replace(filepath: str, old: str, new: str, count: int = 1) -> JsonDict:
    """Edit a file by replacing a substring (safe, deterministic)."""
    path = _resolve_path_in_workspace(filepath)
    if not path.exists() or not path.is_file():
        return {"filepath": filepath, "error": "File not found"}

    text = _read_text(path)
    if old not in text:
        return {"filepath": filepath, "error": "Target string not found"}

    replaced = text.replace(old, new, int(count))
    _write_text(path, replaced)
    return {"filepath": str(path.relative_to(_workspace_root())), "replacements": min(text.count(old), int(count))}


class ToolRegistry:
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def list_tools(self) -> list[JsonDict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            }
            for t in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def execute(self, name: str, args: JsonDict) -> Any:
        tool = self.get(name)
        req = self._permission_request_for_tool(name, args)
        if not self.permission_manager.request_permission(req):
            raise PermissionError(f"Permission denied for {name} on {req.target}")
        return tool.fn(**args)

    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _permission_request_for_tool(self, tool_name: str, args: JsonDict) -> PermissionRequest:
        mapping: dict[str, tuple[OperationType, str, PermissionLevel]] = {
            "list_files": (OperationType.LIST_DIRECTORY, "directory", PermissionLevel.READ),
            "read_file": (OperationType.READ_FILE, "filepath", PermissionLevel.READ),
            "search_files": (OperationType.SEARCH_FILES, "directory", PermissionLevel.READ),
            "write_file": (OperationType.WRITE_FILE, "filepath", PermissionLevel.WRITE),
            "edit_file_replace": (OperationType.EDIT_FILE, "filepath", PermissionLevel.WRITE),
        }
        if tool_name not in mapping:
            return PermissionRequest(
                operation=OperationType.EXECUTE_CODE,
                target="system",
                context={"tool": tool_name, "args": args},
                required_level=PermissionLevel.EXECUTE,
            )
        op, key, level = mapping[tool_name]
        target_raw = str(args.get(key, "."))
        # PermissionManager default policies are based on absolute paths.
        try:
            target_abs = str(_resolve_path_in_workspace(target_raw))
        except Exception:
            # If it doesn't resolve in workspace, still pass raw (will likely be denied).
            target_abs = target_raw
        return PermissionRequest(
            operation=op,
            target=target_abs,
            context={"tool": tool_name, "args": args, "reason": f"Running tool {tool_name}"},
            required_level=level,
        )

    def _register_builtin_tools(self) -> None:
        self._register(
            Tool(
                name="list_files",
                description="List files in a directory (workspace-scoped).",
                schema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "default": "**/*"},
                        "max_results": {"type": "integer", "default": 200},
                    },
                    "required": ["directory"],
                },
                fn=tool_list_files,
            )
        )
        self._register(
            Tool(
                name="read_file",
                description="Read a file (optionally by line range).",
                schema={
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "start_line": {"type": "integer", "default": 1},
                        "end_line": {"type": "integer", "default": 200},
                    },
                    "required": ["filepath"],
                },
                fn=tool_read_file,
            )
        )
        self._register(
            Tool(
                name="search_files",
                description="Search for text in files under a directory.",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "directory": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "default": "**/*.py"},
                        "is_regex": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "default": 20},
                        "context_lines": {"type": "integer", "default": 2},
                    },
                    "required": ["query"],
                },
                fn=tool_search_files,
            )
        )
        self._register(
            Tool(
                name="write_file",
                description="Write a file (creates parents).",
                schema={
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "content": {"type": "string"},
                        "overwrite": {"type": "boolean", "default": False},
                    },
                    "required": ["filepath", "content"],
                },
                fn=tool_write_file,
            )
        )
        self._register(
            Tool(
                name="edit_file_replace",
                description="Edit a file by replacing a substring with another substring.",
                schema={
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "count": {"type": "integer", "default": 1},
                    },
                    "required": ["filepath", "old", "new"],
                },
                fn=tool_edit_file_replace,
            )
        )