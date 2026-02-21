"""Monty-backed code interpreter for the RLM sandbox.

Provides :class:`MontyCodeInterpreter`, a standalone code execution engine that
uses ``pydantic-monty`` to run untrusted Python in a restricted sandbox.

Async external functions (like ``llm_query``) are awaited transparently by the
interpreter, so sandbox code does not need ``await``.  Monty's futures mechanism
(``MontyFutureSnapshot``) is also supported for sandbox code that explicitly
uses ``await`` with Monty's built-in ``asyncio`` support.
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

    Async external functions are automatically detected via
    ``inspect.iscoroutine`` and awaited transparently, so the sandbox
    code does not need to use ``await``.  Concurrency for multiple
    sub-LLM calls is provided by ``llm_query_batched``, which internally
    uses ``asyncio.gather``.

    If the sandbox code *does* use ``await`` (Monty supports the
    ``asyncio`` module), async tools are dispatched as Monty futures
    via ``resume(future=...)`` and resolved when the ``await`` triggers
    a ``MontyFutureSnapshot``.

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
        os_access: pydantic_monty.OsAccess | None = None,
    ) -> None:
        self._tools: dict[str, Callable[..., Any]] = dict(tools) if tools else {}
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._os_access: pydantic_monty.OSAccess | None = os_access
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

        Async external functions (those returning coroutines) are
        awaited transparently so the sandbox code can call them without
        ``await``.  Concurrency for batched LLM queries is handled
        inside ``llm_query_batched`` via ``asyncio.gather``.

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
                external_functions=list(all_tools),
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
            )
        except pydantic_monty.MontyRuntimeError as e:
            raise CodeExecutionError(str(e)) from e

        # Track pending async tasks keyed by Monty call_id
        pending_tasks: dict[int, asyncio.Task[tuple[int, dict[str, Any]]]] = {}

        try:
            # -- step through Monty snapshots --------------------------------
            while not isinstance(progress, pydantic_monty.MontyComplete):
                if isinstance(progress, pydantic_monty.MontySnapshot):
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
                        raise CodeExecutionError(
                            f"Unknown function: {progress.function_name}"
                        )
                    try:
                        result = func(*progress.args, **progress.kwargs)
                        # Await coroutines transparently so the sandbox
                        # does not need to use ``await`` for async tools
                        if inspect.iscoroutine(result):
                            result = await result
                    except Exception as e:
                        raise CodeExecutionError(
                            f"Tool {progress.function_name} failed: {e}"
                        ) from e

                    # Resume Monty with the resolved return value
                    progress = progress.resume(return_value=result)

                elif isinstance(progress, pydantic_monty.MontyFutureSnapshot):
                    # Monty needs one or more future results before it can
                    # continue.  Await whichever finishes first, then resume.
                    current_tasks = [
                        pending_tasks[cid]
                        for cid in progress.pending_call_ids
                        if cid in pending_tasks
                    ]
                    done, _ = await asyncio.wait(
                        current_tasks, return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Collect resolved results and remove from pending
                    results: dict[int, dict[str, Any]] = {}
                    for task in done:
                        cid, ext_result = task.result()
                        results[cid] = ext_result
                        pending_tasks.pop(cid, None)
                    progress = progress.resume(results)  # type: ignore[arg-type]

                else:
                    raise CodeExecutionError(
                        f"Unexpected Monty progress type: {type(progress).__name__}"
                    )
        finally:
            # Cancel any outstanding async tasks on early exit (e.g. SUBMIT)
            for task in pending_tasks.values():
                task.cancel()
            if pending_tasks:
                await asyncio.gather(
                    *pending_tasks.values(), return_exceptions=True
                )

        # Return captured stdout or None
        return "".join(stdout_parts) if stdout_parts else None
