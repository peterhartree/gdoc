"""Model Context Protocol server exposing gdoc subcommands as MCP tools.

`gdoc mcp` speaks MCP over stdio so desktop chat clients (Claude Desktop,
ChatGPT desktop, and anything else that launches a local stdio server) can
drive gdoc directly, instead of gdoc only being reachable from a coding
agent with shell access.

Design notes:

- **No new dependencies.** MCP's stdio transport is newline-delimited
  JSON-RPC 2.0, which is short enough to implement here. Adding an SDK
  would pull a web stack into a CLI whose dependency list is deliberately
  four entries long.
- **Schemas are derived from argparse**, not hand-written, so a new flag on
  `gdoc edit` becomes an MCP tool parameter with no work here. Only the
  command allowlist and the read/write classification below are manual.
- **Commands run in-process** via the same dispatch the CLI uses, with
  stdout captured. Nothing shells out.
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from typing import Any

from gdoc import __version__

# MCP revisions this server has been checked against. An unknown version
# from the client falls back to the newest one we know.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

TOOL_PREFIX = "gdoc_"

# Subcommands exposed as tools, and whether each one only reads.
# Anything absent is deliberately not exposed: `auth` needs an interactive
# browser, `update` mutates the install, `config` is machine-wide, and
# `pull`/`push`/`export` work on local file paths a chat client cannot see.
EXPOSED_COMMANDS: dict[str, bool] = {
    # read-only
    "ls": True,
    "find": True,
    "cat": True,
    "info": True,
    "tabs": True,
    "toc": True,
    "cells": True,
    "revisions": True,
    "comments": True,
    "comment-info": True,
    "diff": True,
    "images": True,
    "structure": True,
    "drives": True,
    # writes
    "edit": False,
    "insert": False,
    "write": False,
    "add-tab": False,
    "insert-image": False,
    "replace-image": False,
    "comment": False,
    "reply": False,
    "resolve": False,
    "reopen": False,
    "delete-comment": False,
    "new": False,
    "cp": False,
    "mkdir": False,
    "mv": False,
    "rename": False,
    "share": False,
}

# Commands whose content argument is a local markdown file. A chat client
# has no filesystem to write one to, so these tools also accept inline
# `text`, which the server materialises to a temp file for the duration of
# the call. The CLI itself is unchanged.
_TEXT_TO_FILE: dict[str, str] = {
    "write": "file",
    "insert": "file",
}

_TEXT_DESCRIPTION = (
    "Markdown content, supplied inline. Use this instead of `file` when you "
    "have no local filesystem. Exactly one of `text` or `file` is required."
)

# Parser-level plumbing that must not become a tool parameter.
_SKIP_DESTS = frozenset({
    "help",
    "func",
    "command",
    "allow_commands",
    "verbose",
    "plain",
})


def _tool_name(command: str) -> str:
    return TOOL_PREFIX + command.replace("-", "_")


def _command_name(tool: str) -> str:
    if not tool.startswith(TOOL_PREFIX):
        raise KeyError(tool)
    return tool[len(TOOL_PREFIX):].replace("_", "-")


def _help_text(action: argparse.Action) -> str:
    """Render an action's help, tolerating argparse's %(default)s syntax."""
    help_str = action.help or ""
    if "%" not in help_str:
        return help_str
    try:
        return help_str % {"default": action.default, "prog": "gdoc"}
    except (KeyError, TypeError, ValueError):
        return help_str


def _property_for(action: argparse.Action) -> dict[str, Any]:
    """Map one argparse action onto a JSON Schema property."""
    prop: dict[str, Any] = {}
    description = _help_text(action)

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        prop["type"] = "boolean"
    elif action.choices:
        prop["type"] = "string"
        prop["enum"] = [str(c) for c in action.choices]
    elif action.nargs in ("+", "*") or isinstance(action, argparse._AppendAction):
        prop["type"] = "array"
        prop["items"] = {"type": "integer" if action.type is int else "string"}
    elif action.type is int:
        prop["type"] = "integer"
    elif action.type is float:
        prop["type"] = "number"
    else:
        prop["type"] = "string"

    if description:
        prop["description"] = description
    return prop


def _schema_for(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Derive a JSON Schema for a subparser's arguments."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for action in parser._actions:
        if action.dest in _SKIP_DESTS:
            continue
        if isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue

        properties[action.dest] = _property_for(action)
        is_positional = not action.option_strings
        if is_positional and action.nargs not in ("?", "*"):
            required.append(action.dest)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _description_for(command: str, parser: argparse.ArgumentParser) -> str:
    """Prefer the subparser's long description, falling back to its help."""
    text = (parser.description or "").strip()
    if not text:
        text = (getattr(parser, "_gdoc_help", "") or "").strip()
    if not EXPOSED_COMMANDS[command]:
        warning = "Writes to Google Docs/Drive."
        text = f"{text}\n\n{warning}" if text else warning
    return text


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def build_tools(
    *, read_only: bool = False, allow: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build the MCP tool list by introspecting the gdoc CLI parser.

    Returns a mapping of tool name -> tool definition, in the order the
    commands are listed in EXPOSED_COMMANDS.
    """
    from gdoc.cli import build_parser

    parser = build_parser()
    subparsers = _subparsers(parser)
    help_by_command = _help_by_command(parser)

    tools: dict[str, dict[str, Any]] = {}
    for command, is_read_only in EXPOSED_COMMANDS.items():
        if command not in subparsers:
            continue  # command retired upstream; skip rather than crash
        if read_only and not is_read_only:
            continue
        if allow is not None and command not in allow:
            continue

        sub = subparsers[command]
        sub._gdoc_help = help_by_command.get(command, "")
        schema = _schema_for(sub)

        file_arg = _TEXT_TO_FILE.get(command)
        if file_arg and file_arg in schema["properties"]:
            schema["properties"]["text"] = {
                "type": "string",
                "description": _TEXT_DESCRIPTION,
            }
            schema["required"] = [
                r for r in schema.get("required", []) if r != file_arg
            ]
            if not schema["required"]:
                del schema["required"]

        tools[_tool_name(command)] = {
            "name": _tool_name(command),
            "description": _description_for(command, sub),
            "inputSchema": schema,
        }
    return tools


def _help_by_command(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Recover each subcommand's one-line help from the subparsers action."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return {
                choice.dest: (choice.help or "")
                for choice in action._choices_actions
            }
    return {}


def _argv_for(
    command: str, arguments: dict[str, Any], parser: argparse.ArgumentParser
) -> list[str]:
    """Turn a tool-call argument dict back into a gdoc argv list."""
    actions = {
        a.dest: a
        for a in parser._actions
        if a.dest not in _SKIP_DESTS
        and not isinstance(a, (argparse._HelpAction, argparse._VersionAction))
    }

    unknown = set(arguments) - set(actions)
    if unknown:
        raise ValueError(f"unknown argument(s): {', '.join(sorted(unknown))}")

    positionals: list[str] = []
    options: list[str] = []

    for dest, action in actions.items():
        if dest not in arguments:
            continue
        value = arguments[dest]
        if value is None:
            continue

        if not action.option_strings:
            if isinstance(value, list):
                positionals.extend(str(v) for v in value)
            else:
                positionals.append(str(value))
            continue

        flag = max(action.option_strings, key=len)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if value:
                options.append(flag)
        elif isinstance(action, argparse._AppendAction) and isinstance(value, list):
            for item in value:
                options.extend([flag, str(item)])
        elif isinstance(value, list):
            options.append(flag)
            options.extend(str(v) for v in value)
        else:
            options.extend([flag, str(value)])

    return [command, *options, *positionals]


def call_command(
    command: str, arguments: dict[str, Any]
) -> tuple[str, str, int]:
    """Run a gdoc subcommand in-process, capturing its output.

    Returns (stdout, stderr, exit_code).
    """
    from gdoc.cli import build_parser, run_argv

    parser = build_parser()
    subparser = _subparsers(parser).get(command)
    if subparser is None:
        raise ValueError(f"unknown command: {command}")

    _reset_account_state(arguments.get("account"))

    with _materialised_text(command, arguments) as prepared:
        argv = _argv_for(command, prepared, subparser)

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = run_argv(argv, check_updates=False)
            except SystemExit as e:  # argparse usage errors exit, not raise
                code = e.code if isinstance(e.code, int) else 1
    return out.getvalue(), err.getvalue(), code


def _reset_account_state(account: str | None) -> None:
    """Make each tool call behave like a fresh CLI invocation.

    `set_active_account()` is process-global and the API service objects
    are `lru_cache`d on the assumption of one account per process — true
    for the CLI, false for a long-lived server. Without this, a call that
    names an account would leak into every later call, and cached service
    objects would keep using the first account's credentials.
    """
    from gdoc.util import get_active_account, set_active_account

    if account == get_active_account():
        return

    from gdoc.api import get_drive_service, get_sheets_service
    from gdoc.api.docs import get_docs_service
    from gdoc.api.revisions import _get_session

    for cached in (
        get_drive_service, get_sheets_service, get_docs_service, _get_session,
    ):
        cached.cache_clear()
    set_active_account(account)


@contextlib.contextmanager
def _materialised_text(command: str, arguments: dict[str, Any]):
    """Swap an inline `text` argument for a temp file the CLI can read."""
    file_arg = _TEXT_TO_FILE.get(command)
    if file_arg is None or "text" not in arguments:
        yield arguments
        return

    if arguments.get(file_arg):
        raise ValueError(f"pass either `text` or `{file_arg}`, not both")

    prepared = {k: v for k, v in arguments.items() if k != "text"}
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", encoding="utf-8", delete=True
    ) as handle:
        handle.write(arguments["text"])
        handle.flush()
        prepared[file_arg] = handle.name
        yield prepared


def _clean_notes(stderr: str) -> str:
    """Drop pre-flight banners that say nothing happened.

    `pre_flight()` writes a change summary to stderr before most commands.
    Real changes are worth relaying to the model; "no changes" is noise on
    every single tool call.
    """
    lines = [
        line for line in stderr.splitlines()
        if line.strip() not in ("--- no changes ---", "---", "")
    ]
    return "\n".join(lines).strip()


class MCPServer:
    """Minimal MCP server over newline-delimited JSON-RPC on stdio."""

    def __init__(
        self,
        *,
        read_only: bool = False,
        allow: set[str] | None = None,
        account: str | None = None,
    ) -> None:
        self.tools = build_tools(read_only=read_only, allow=allow)
        self.account = account
        self.protocol_version = DEFAULT_PROTOCOL_VERSION

    # -- request handlers ------------------------------------------------

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        if requested in SUPPORTED_PROTOCOL_VERSIONS:
            self.protocol_version = requested
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "gdoc", "version": __version__},
            "instructions": (
                "gdoc reads and edits Google Docs and Drive files. Pass a "
                "document URL or bare ID as `doc`. Output is terse by design; "
                "set `json` for machine-readable output."
            ),
        }

    def handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": list(self.tools.values())}

    def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if name not in self.tools:
            return self._text_result(f"ERR: no such tool: {name}", is_error=True)

        if self.account and "account" not in arguments:
            arguments = {**arguments, "account": self.account}

        try:
            stdout, stderr, code = call_command(_command_name(name), arguments)
        except Exception as e:  # surface as a tool error, never kill the server
            return self._text_result(f"ERR: {e}", is_error=True)

        if code != 0:
            body = (stderr or stdout).strip() or f"exit code {code}"
            return self._text_result(body, is_error=True)

        text = stdout.strip()
        notes = _clean_notes(stderr)
        if notes:
            text = f"{text}\n\n--- notes ---\n{notes}" if text else notes
        return self._text_result(text or "OK")

    @staticmethod
    def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }

    # -- dispatch --------------------------------------------------------

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        params = message.get("params") or {}
        msg_id = message.get("id")
        is_notification = "id" not in message

        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
            "ping": lambda _params: {},
        }

        if method in ("notifications/initialized", "notifications/cancelled"):
            return None

        handler = handlers.get(method)
        if handler is None:
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }

        try:
            result = handler(params)
        except Exception as e:
            if is_notification:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)},
            }

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    # -- transport -------------------------------------------------------

    def serve(self, stdin=None, stdout=None) -> int:
        stdin = stdin or sys.stdin
        protocol_out = stdout or sys.stdout

        # Anything that slips a print() past the per-call capture must not
        # corrupt the JSON-RPC stream.
        real_stdout, sys.stdout = sys.stdout, sys.stderr
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._write(
                        protocol_out,
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": "parse error"},
                        },
                    )
                    continue

                messages = message if isinstance(message, list) else [message]
                for item in messages:
                    if not isinstance(item, dict):
                        continue
                    response = self.dispatch(item)
                    if response is not None:
                        self._write(protocol_out, response)
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        finally:
            sys.stdout = real_stdout
        return 0

    @staticmethod
    def _write(stream, payload: dict[str, Any]) -> None:
        stream.write(json.dumps(payload) + "\n")
        stream.flush()
