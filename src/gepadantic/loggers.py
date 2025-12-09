"""Rich console logger with enhanced formatting for GEPA optimization messages.

This module provides a custom logger that uses Rich to format and highlight
messages during GEPA optimization, with special formatting for proposal messages.
"""

import re
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from gepa.logging.logger import LoggerProtocol


class RichConsoleLogger(LoggerProtocol):
    """Logger that logs to the console using rich with enhanced formatting.
    
    This logger provides special formatting for "Proposed new text" messages
    that appear during GEPA optimization iterations. These messages are
    highlighted in a panel with a clean header showing the iteration number
    and component name.
    
    Example:
        ```python
        from gepadantic.helpers import RichConsoleLogger
        
        logger = RichConsoleLogger()
        logger.log("Normal message")
        logger.log("Iteration 1: Proposed new text for instructions: ...")
        # Displays as a panel with title "Iteration 1: instructions"
        ```
    """
    
    def __init__(
        self, 
        proposal_color: str = "cyan",
        proposal_border_style: str = "bold cyan",
        console: Console | None = None,
    ) -> None:
        """Initialize the Rich console logger.
        
        Args:
            proposal_color: Color to use for proposal message text. 
                Default is "cyan". Common options: "cyan", "magenta", "yellow", 
                "green", "blue", etc.
            proposal_border_style: Style for the panel border around proposals.
                Default is "bold cyan". Can include styles like "bold", "dim", etc.
            console: Console instance to use for logging. If None, creates a new 
                Console instance with safe defaults.
        """
        self.proposal_color = proposal_color
        self.proposal_border_style = proposal_border_style
        # Create console with force_terminal=True to handle tqdm interactions better
        self.console = console if console is not None else Console(force_terminal=True)
    
    def _parse_proposal_message(self, message: str) -> Optional[tuple[str, str, str]]:
        """Parse a proposal message to extract its components.
        
        Proposal messages follow this pattern:
        "Iteration N: Proposed new text for <component>: <content>"
        
        Args:
            message: The log message to parse.
            
        Returns:
            A tuple of (iteration, component, content) if this is a proposal message,
            None otherwise.
            
        Example:
            >>> logger._parse_proposal_message(
            ...     "Iteration 1: Proposed new text for instructions: Do this..."
            ... )
            ("Iteration 1", "instructions", "Do this...")
        """
        # Pattern: "Iteration N: Proposed new text for <component>: <content>"
        pattern = r'^(Iteration\s+\d+):\s+Proposed new text for\s+(.+?):\s+(.*)$'
        match = re.match(pattern, message, re.DOTALL)
        
        if match:
            iteration = match.group(1)
            component = match.group(2)
            content = match.group(3)
            return (iteration, component, content)
        
        return None
    
    def log(self, message: str) -> None:
        """Log a message to the console with enhanced formatting.
        
        If the message is a "Proposed new text" message, it will be displayed
        in a highlighted panel with special formatting. Otherwise, it will be
        printed normally.
        
        Note:
            When used alongside tqdm progress bars, this logger ensures messages
            appear on fresh lines to prevent overlap with progress bar output.
        
        Args:
            message: The message to log.
        """
        
        parsed = self._parse_proposal_message(message)
        
        if parsed:
            iteration, component, content = parsed
            
            # Create a clean, simple title: "Iteration N: Component"
            title_text = Text()
            title_text.append(f"{iteration}: ", style="bold white")
            title_text.append(component, style=f"bold {self.proposal_color}")
            
            # Create the panel with the content
            # Truncate very long content for readability in the panel
            display_content = content
            if len(content) > 2000:
                display_content = content[:2000] + "\n... [dim](content truncated)[/dim]"
            
            panel = Panel(
                display_content,
                title=title_text,
                border_style=self.proposal_border_style,
                padding=(1, 2),
            )
            
            self.console.print(panel)
        else:
            # Regular message - print normally
            self.console.print(message)
            
    def render_optimization_dag(self, dot_string: str, copy_to_clipboard: bool = False) -> None:
        """Render an optimization DAG graph in the console.
        
        Displays the graph with syntax highlighting and metadata summary.
        
        Args:
            dot_string: Graphviz DOT format string of the DAG.
            copy_to_clipboard: Whether to copy the DOT string to clipboard.
            
        Example:
            >>> logger = RichConsoleLogger()
            >>> logger.render_optimization_dag(result.graphviz_dag)
        """
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.console import Group
        import re
        
        # Create metadata table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Count nodes and edges
        nodes = len(re.findall(r'^\s*\d+\s+\[', dot_string, re.MULTILINE))
        edges = len(re.findall(r'\d+\s*->\s*\d+', dot_string))
        
        # Find best and dominators
        best_match = re.search(r'(\d+)\s+\[label="([^"]+)".*fillcolor=cyan', dot_string)
        best_node = f"Node {best_match.group(1)}" if best_match else "N/A"
        
        dominators = len(re.findall(r'fillcolor=orange', dot_string))
        
        table.add_row("Programs", str(nodes))
        table.add_row("Evolutions", str(edges))
        table.add_row("Best", best_node)
        table.add_row("Dominators", str(dominators))
        
        # Syntax highlighted DOT
        syntax = Syntax(dot_string, "dot", theme="monokai", line_numbers=False)
        
        # Group and panel
        group = Group(table, "", syntax)
        panel = Panel(
            group, 
            title="[bold cyan]Optimization Evolution DAG[/bold cyan]", 
            border_style=self.proposal_border_style,
            padding=(1, 2)
        )
        
        self.console.print(panel)
        
            # Handle clipboard copying
        if copy_to_clipboard:
            try:
                import pyperclip
                pyperclip.copy(dot_string)
                self.console.print("[green]✓[/green] DOT string copied to clipboard!")
            except ImportError:
                self.console.print("[yellow]⚠[/yellow] pyperclip not installed. Install with: pip install pyperclip")
            except Exception as e:
                self.console.print(f"[yellow]⚠[/yellow] Could not copy to clipboard: {e}")