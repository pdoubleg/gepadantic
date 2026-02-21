"""Tests for the MontyCodeInterpreter sandbox.

These tests exercise the interpreter in isolation -- no LLM calls are made.
They verify that code execution, variable injection, SAVE/CLEAR/SUBMIT
semantics, error handling, and custom tools all behave as expected.
"""

from __future__ import annotations

import pytest

from gepadantic.rlm.interpreter import MontyCodeInterpreter
from gepadantic.rlm.types import CodeExecutionError, FinalOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def interp() -> MontyCodeInterpreter:
    """Return a fresh interpreter with type-checking disabled."""
    return MontyCodeInterpreter(type_check=False)


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    """Tests for print output and return values."""

    @pytest.mark.asyncio
    async def test_basic_print(self, interp: MontyCodeInterpreter) -> None:
        """Executing ``print('hello')`` should capture 'hello' in output."""
        result = await interp.execute_async("print('hello')")
        assert result is not None
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_multiple_prints(self, interp: MontyCodeInterpreter) -> None:
        """Multiple ``print()`` calls should all appear in the output."""
        result = await interp.execute_async("print('aaa')\nprint('bbb')")
        assert result is not None
        assert "aaa" in result
        assert "bbb" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_none(self, interp: MontyCodeInterpreter) -> None:
        """Code that produces no print output should return ``None``."""
        result = await interp.execute_async("x = 42")
        assert result is None

    @pytest.mark.asyncio
    async def test_expression_no_output(self, interp: MontyCodeInterpreter) -> None:
        """A bare expression (no print) should still return ``None``."""
        result = await interp.execute_async("1 + 1")
        assert result is None


# ---------------------------------------------------------------------------
# Variable injection
# ---------------------------------------------------------------------------


class TestVariableInjection:
    """Tests for injecting variables into the sandbox namespace."""

    @pytest.mark.asyncio
    async def test_variables_injected(self, interp: MontyCodeInterpreter) -> None:
        """Injected variables should be accessible inside the sandbox."""
        result = await interp.execute_async(
            "print(x + y)",
            variables={"x": 10, "y": 32},
        )
        assert result is not None
        assert "42" in result

    @pytest.mark.asyncio
    async def test_string_variable(self, interp: MontyCodeInterpreter) -> None:
        """String variables should be passed through intact."""
        result = await interp.execute_async(
            "print(greeting)",
            variables={"greeting": "hello world"},
        )
        assert result is not None
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_list_variable(self, interp: MontyCodeInterpreter) -> None:
        """List variables should be accessible and iterable."""
        result = await interp.execute_async(
            "print(len(items))",
            variables={"items": [1, 2, 3, 4, 5]},
        )
        assert result is not None
        assert "5" in result

    @pytest.mark.asyncio
    async def test_dict_variable(self, interp: MontyCodeInterpreter) -> None:
        """Dict variables should be accessible with key lookup."""
        result = await interp.execute_async(
            "print(data['key'])",
            variables={"data": {"key": "value123"}},
        )
        assert result is not None
        assert "value123" in result

    @pytest.mark.asyncio
    async def test_empty_variables(self, interp: MontyCodeInterpreter) -> None:
        """Passing empty variables dict should not break execution."""
        result = await interp.execute_async("print('ok')", variables={})
        assert result is not None
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_none_variables(self, interp: MontyCodeInterpreter) -> None:
        """Passing ``None`` for variables should not break execution."""
        result = await interp.execute_async("print('ok')", variables=None)
        assert result is not None
        assert "ok" in result


# ---------------------------------------------------------------------------
# SAVE / CLEAR persistence
# ---------------------------------------------------------------------------


class TestSaveClear:
    """Tests for SAVE() and CLEAR() state persistence across calls."""

    @pytest.mark.asyncio
    async def test_save_persists_across_executions(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """A SAVE'd variable should be available in a subsequent execute_async()."""
        await interp.execute_async("SAVE(answer=42)")
        result = await interp.execute_async("print(answer)")
        assert result is not None
        assert "42" in result

    @pytest.mark.asyncio
    async def test_save_multiple_variables(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """Multiple variables can be SAVE'd at once."""
        await interp.execute_async("SAVE(a=1, b=2)")
        result = await interp.execute_async("print(a + b)")
        assert result is not None
        assert "3" in result

    @pytest.mark.asyncio
    async def test_save_overwrites(self, interp: MontyCodeInterpreter) -> None:
        """SAVE with the same key should overwrite the old value."""
        await interp.execute_async("SAVE(val=10)")
        await interp.execute_async("SAVE(val=99)")
        result = await interp.execute_async("print(val)")
        assert result is not None
        assert "99" in result

    @pytest.mark.asyncio
    async def test_clear_removes_saved(self, interp: MontyCodeInterpreter) -> None:
        """CLEAR('name') should remove that specific saved variable."""
        await interp.execute_async("SAVE(keep=1, remove=2)")
        await interp.execute_async("CLEAR('remove')")
        result = await interp.execute_async("print(keep)")
        assert result is not None
        assert "1" in result

    @pytest.mark.asyncio
    async def test_clear_all(self, interp: MontyCodeInterpreter) -> None:
        """CLEAR() with no args should remove all saved state."""
        await interp.execute_async("SAVE(a=1, b=2, c=3)")
        await interp.execute_async("CLEAR()")
        # After clearing, previously saved variables should not exist
        with pytest.raises((CodeExecutionError, NameError)):
            await interp.execute_async("print(a)")

    @pytest.mark.asyncio
    async def test_clear_nonexistent_is_safe(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """CLEAR('nonexistent') should not raise an error."""
        await interp.execute_async("CLEAR('ghost')")

    @pytest.mark.asyncio
    async def test_save_return_message(self, interp: MontyCodeInterpreter) -> None:
        """SAVE should return a confirmation message via the tool."""
        result = await interp.execute_async("result = SAVE(x=1)\nprint(result)")
        assert result is not None
        assert "Saved" in result


# ---------------------------------------------------------------------------
# SUBMIT / FinalOutput
# ---------------------------------------------------------------------------


class TestSubmit:
    """Tests for SUBMIT() producing FinalOutput sentinel values."""

    @pytest.mark.asyncio
    async def test_submit_returns_final_output(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """SUBMIT(key=value) should return a FinalOutput instance."""
        interp.output_fields = [{"name": "answer"}]
        result = await interp.execute_async("SUBMIT(answer='hello')")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "hello"}

    @pytest.mark.asyncio
    async def test_submit_multiple_fields(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """SUBMIT with multiple keyword args should capture all of them."""
        interp.output_fields = [{"name": "a"}, {"name": "b"}]
        result = await interp.execute_async("SUBMIT(a=1, b=2)")
        assert isinstance(result, FinalOutput)
        assert result.output["a"] == 1
        assert result.output["b"] == 2

    @pytest.mark.asyncio
    async def test_submit_positional_with_output_fields(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """Positional SUBMIT args should map to output_fields names in order."""
        interp.output_fields = [{"name": "first"}, {"name": "second"}]
        result = await interp.execute_async("SUBMIT('alpha', 'beta')")
        assert isinstance(result, FinalOutput)
        assert result.output["first"] == "alpha"
        assert result.output["second"] == "beta"

    @pytest.mark.asyncio
    async def test_submit_without_output_fields(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """SUBMIT with kwargs but no output_fields should still capture kwargs."""
        interp.output_fields = None
        result = await interp.execute_async("SUBMIT(answer='test')")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "test"}

    @pytest.mark.asyncio
    async def test_submit_stops_execution(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """Code after SUBMIT should not execute (SUBMIT is immediate)."""
        interp.output_fields = [{"name": "val"}]
        result = await interp.execute_async("SUBMIT(val=1)\nprint('should not appear')")
        # SUBMIT is intercepted at the snapshot level, so subsequent code
        # never runs.
        assert isinstance(result, FinalOutput)
        assert result.output["val"] == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for syntax and runtime error propagation."""

    @pytest.mark.asyncio
    async def test_syntax_error_raises(self, interp: MontyCodeInterpreter) -> None:
        """Malformed Python should raise SyntaxError."""
        with pytest.raises(SyntaxError):
            await interp.execute_async("def foo(:")

    @pytest.mark.asyncio
    async def test_runtime_error_raises(self, interp: MontyCodeInterpreter) -> None:
        """Runtime errors (e.g. ZeroDivisionError) should raise CodeExecutionError."""
        with pytest.raises(CodeExecutionError):
            await interp.execute_async("x = 1 / 0")

    @pytest.mark.asyncio
    async def test_name_error_raises(self, interp: MontyCodeInterpreter) -> None:
        """Referencing an undefined name should raise CodeExecutionError."""
        with pytest.raises(CodeExecutionError):
            await interp.execute_async("print(undefined_var)")

    @pytest.mark.asyncio
    async def test_unknown_function_raises(
        self,
        interp: MontyCodeInterpreter,
    ) -> None:
        """Calling a function not in tools should raise CodeExecutionError."""
        with pytest.raises((CodeExecutionError, SyntaxError)):
            await interp.execute_async("unknown_func()")


# ---------------------------------------------------------------------------
# Custom tools
# ---------------------------------------------------------------------------


class TestCustomTools:
    """Tests for user-provided tool functions."""

    @pytest.mark.asyncio
    async def test_custom_tool_callable(self) -> None:
        """A custom tool should be callable from sandbox code."""

        def my_tool(text: str) -> str:
            return f"processed: {text}"

        interp = MontyCodeInterpreter(
            tools={"my_tool": my_tool},
            type_check=False,
        )
        result = await interp.execute_async("result = my_tool('hello')\nprint(result)")
        assert result is not None
        assert "processed: hello" in result

    @pytest.mark.asyncio
    async def test_multiple_custom_tools(self) -> None:
        """Multiple custom tools should all be accessible."""

        def tool_a() -> str:
            return "from_a"

        def tool_b() -> str:
            return "from_b"

        interp = MontyCodeInterpreter(
            tools={"tool_a": tool_a, "tool_b": tool_b},
            type_check=False,
        )
        result = await interp.execute_async("print(tool_a())\nprint(tool_b())")
        assert result is not None
        assert "from_a" in result
        assert "from_b" in result

    @pytest.mark.asyncio
    async def test_tool_with_kwargs(self) -> None:
        """Tools receiving keyword arguments should work."""

        def adder(a: int, b: int) -> str:
            return str(a + b)

        interp = MontyCodeInterpreter(
            tools={"adder": adder},
            type_check=False,
        )
        result = await interp.execute_async("print(adder(a=3, b=7))")
        assert result is not None
        assert "10" in result

    @pytest.mark.asyncio
    async def test_async_tool_dispatched_as_future(self) -> None:
        """An async tool should be dispatched as a Monty future and resolved."""

        async def async_greet(name: str) -> str:
            return f"hello {name}"

        interp = MontyCodeInterpreter(
            tools={"async_greet": async_greet},
            type_check=False,
        )
        result = await interp.execute_async(
            "msg = async_greet('world')\nprint(msg)"
        )
        assert result is not None
        assert "hello world" in result


# ---------------------------------------------------------------------------
# Lifecycle: start() / shutdown()
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests for interpreter lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_clears_state(self, interp: MontyCodeInterpreter) -> None:
        """Calling start() should clear all SAVE'd state."""
        await interp.execute_async("SAVE(secret=123)")
        interp.start()
        # After start(), saved state should be gone
        with pytest.raises((CodeExecutionError, NameError)):
            await interp.execute_async("print(secret)")

    def test_shutdown_is_noop(self, interp: MontyCodeInterpreter) -> None:
        """shutdown() should not raise (it's a no-op for Monty)."""
        interp.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_reuse_after_start(self, interp: MontyCodeInterpreter) -> None:
        """Interpreter should be reusable after start() is called."""
        await interp.execute_async("SAVE(val=1)")
        interp.start()
        await interp.execute_async("SAVE(val=2)")
        result = await interp.execute_async("print(val)")
        assert result is not None
        assert "2" in result
