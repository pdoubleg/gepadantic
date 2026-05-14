"""Monty-backed code interpreter for the RLM sandbox.

Provides :class:`MontyCodeInterpreter`, a standalone code execution engine that
uses ``pydantic-monty`` to run untrusted Python in a restricted sandbox.

Async external functions (like ``llm_query``) must be awaited by sandbox code.
The interpreter resolves Monty's future snapshots so multiple awaited tool calls
can run concurrently.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import pydantic_monty

from .types import CodeExecutionError, FinalOutput


class MontyCodeInterpreter:
    """Code interpreter backed by the ``pydantic-monty`` restricted sandbox.

    The interpreter manages a persistent ``_state`` dict so that values
    persisted via ``SAVE()`` survive across successive ``execute_async()``
    calls.  ``CLEAR()`` removes saved values and ``SUBMIT()`` signals that
    the RLM loop should terminate with a final answer.

    Async external functions must be called with ``await``.  When a tool
    returns an awaitable, it is dispatched as a Monty future via
    ``resume({"future": ...})`` and resolved when Monty yields a
    ``FutureSnapshot``.

    Args:
        tools: Dictionary mapping tool names to callable functions.
               Tools may be sync or async.  Async tools are awaited
               before resuming Monty execution.
        type_check: Whether to type-check the code by default.
        type_check_stubs: Optional code to prepend before type-checking
            to serve as stubs.
        limits: Optional resource limits for code execution.

    Example:
        >>> interp = MontyCodeInterpreter()
        >>> result = await interp.execute_async(
        ...     "print(x + y)", variables={"x": 1, "y": 2}
        ... )
        >>> assert "3" in result
    """

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        type_check: bool = True,
        type_check_stubs: str | None = None,
        limits: pydantic_monty.ResourceLimits | None = None,
        os_access: pydantic_monty.OSAccess | None = None,
    ) -> None:
        self._tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._os_access = os_access
        # Persistent state preserved across execute_async() calls via SAVE/CLEAR
        self._state: dict[str, Any] = {}
        # Output field metadata for positional-arg SUBMIT mapping
        self.output_fields: list[dict[str, Any]] | None = None

    # -- public properties --------------------------------------------------

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        """Return the mutable tools dictionary (allows external injection)."""
        return self._tools

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Reset persistent sandbox state.

        Call before a fresh RLM run to ensure no stale ``SAVE``-d variables
        leak across invocations.
        """
        self._state.clear()

    def shutdown(self) -> None:
        """Clean up resources (no-op for Monty, always shuts down)."""

    # -- execution ----------------------------------------------------------

    async def execute_async(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute *code* inside the Monty sandbox and return the result.

        Async external functions (those returning coroutines) must be
        awaited by sandbox code.  The interpreter schedules those
        coroutines as Monty futures and resolves them when execution
        reaches a ``FutureSnapshot``.

        Args:
            code: Python source to execute.
            variables: Variables to inject into the sandbox namespace.  These
                are merged with any values persisted via ``SAVE()``.

        Returns:
            * ``FinalOutput`` if the code called ``SUBMIT()``.
            * A string of captured ``print()`` output.
            * ``None`` if the code produced no output.

        Raises:
            SyntaxError: If the code has a syntax error.
            CodeExecutionError: If the code fails at runtime, a tool call
                fails, or an unsupported Monty state is encountered.
        """
        # Build the full tools dict including built-in SAVE / CLEAR / SUBMIT
        all_tools = dict(self._tools)

        def _save(**kwargs: Any) -> str:
            """Persist variables across iterations."""
            self._state.update(kwargs)
            return f"Saved: {', '.join(kwargs.keys())}"

        def _clear(*names: str) -> str:
            """Remove saved variables (all if no args)."""
            if not names:
                self._state.clear()
                return "Cleared all saved state"
            for n in names:
                self._state.pop(n, None)
            return f"Cleared: {', '.join(names)}"

        all_tools["SAVE"] = _save
        all_tools["CLEAR"] = _clear
        # SUBMIT is a sentinel -- we intercept it in the progress loop below
        all_tools["SUBMIT"] = lambda **kwargs: None

        # Merge caller-provided variables with persisted state
        merged_vars: dict[str, Any] = {}
        if variables:
            merged_vars.update(variables)
        merged_vars.update(self._state)

        # -- compile ---------------------------------------------------------
        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=list(merged_vars) if merged_vars else [],
                type_check=self._type_check,
                type_check_stubs=self._type_check_stubs,
            )
        except pydantic_monty.MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except pydantic_monty.MontyTypingError as e:
            raise CodeExecutionError(str(e)) from e

        # -- execute with stdout capture -------------------------------------
        stdout_parts: list[str] = []

        def _capture_print(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        try:
            progress = monty.start(
                inputs=merged_vars or None,
                limits=self._limits,
                print_callback=_capture_print,
                os=self._os_access,
            )
        except pydantic_monty.MontyRuntimeError as e:
            raise CodeExecutionError(str(e)) from e

        # Track pending async tasks keyed by Monty call_id
        pending_tasks: dict[int, asyncio.Task[Any]] = {}

        try:
            # -- step through Monty snapshots --------------------------------
            while not isinstance(progress, pydantic_monty.MontyComplete):
                if isinstance(progress, pydantic_monty.NameLookupSnapshot):
                    progress = progress.resume(os=self._os_access)

                elif isinstance(progress, pydantic_monty.FunctionSnapshot):
                    # Intercept SUBMIT to signal final output
                    if progress.function_name == "SUBMIT":
                        submit_kwargs = dict(progress.kwargs)
                        # Map positional args to output field names
                        if progress.args and self.output_fields:
                            field_names = [f["name"] for f in self.output_fields]
                            for name, value in zip(field_names, progress.args):
                                submit_kwargs.setdefault(name, value)
                        return FinalOutput(submit_kwargs)

                    # Dispatch to the appropriate tool function
                    func = all_tools.get(progress.function_name)
                    if func is None:
                        progress = progress.resume(
                            {
                                "exception": NameError(
                                    f"Unknown function: {progress.function_name}"
                                )
                            },
                            os=self._os_access,
                        )
                        continue
                    try:
                        result = func(*progress.args, **progress.kwargs)
                        if inspect.isawaitable(result):
                            pending_tasks[progress.call_id] = asyncio.ensure_future(
                                result
                            )
                            progress = progress.resume(
                                {"future": ...},
                                os=self._os_access,
                            )
                        else:
                            progress = progress.resume(
                                {"return_value": result},
                                os=self._os_access,
                            )
                    except Exception as e:
                        progress = progress.resume(
                            {"exception": e},
                            os=self._os_access,
                        )

                elif isinstance(progress, pydantic_monty.FutureSnapshot):
                    # Monty needs one or more future results before it can
                    # continue. Await the requested calls together so independent
                    # async tools can make progress concurrently.
                    results: dict[int, pydantic_monty.ExternalResult] = {}
                    gather_ids: list[int] = []
                    gather_tasks: list[asyncio.Task[Any]] = []

                    for cid in progress.pending_call_ids:
                        task = pending_tasks.pop(cid, None)
                        if task is None:
                            results[cid] = pydantic_monty.ExternalException(
                                exception=RuntimeError(
                                    f"No pending task for Monty future call_id={cid}"
                                )
                            )
                        else:
                            gather_ids.append(cid)
                            gather_tasks.append(task)

                    if gather_tasks:
                        settled = await asyncio.gather(
                            *gather_tasks,
                            return_exceptions=True,
                        )
                        for cid, outcome in zip(gather_ids, settled):
                            if isinstance(outcome, Exception):
                                results[cid] = pydantic_monty.ExternalException(
                                    exception=outcome,
                                )
                            elif isinstance(outcome, BaseException):
                                raise outcome
                            else:
                                results[cid] = pydantic_monty.ExternalReturnValue(
                                    return_value=outcome,
                                )

                    progress = progress.resume(results, os=self._os_access)

                else:
                    raise CodeExecutionError(
                        f"Unexpected Monty progress type: {type(progress).__name__}"
                    )
        except pydantic_monty.MontyRuntimeError as e:
            raise CodeExecutionError(str(e)) from e
        finally:
            # Cancel any outstanding async tasks on early exit (e.g. SUBMIT)
            for task in pending_tasks.values():
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks.values(), return_exceptions=True)

        # Return captured stdout or None
        return "".join(stdout_parts) if stdout_parts else None
