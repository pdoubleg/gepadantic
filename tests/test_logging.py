"""Unit tests for the enhanced Rich console logger."""

from gepadantic.loggers import RichConsoleLogger


class TestRichConsoleLogger:
    """Test suite for RichConsoleLogger."""
    
    def test_init_default_colors(self):
        """Test logger initialization with default colors."""
        logger = RichConsoleLogger()
        assert logger.proposal_color == "cyan"
        assert logger.proposal_border_style == "bold cyan"
    
    def test_init_custom_colors(self):
        """Test logger initialization with custom colors."""
        logger = RichConsoleLogger(
            proposal_color="magenta",
            proposal_border_style="bold magenta"
        )
        assert logger.proposal_color == "magenta"
        assert logger.proposal_border_style == "bold magenta"
    
    def test_parse_simple_proposal(self):
        """Test parsing a simple proposal message."""
        logger = RichConsoleLogger()
        message = "Iteration 1: Proposed new text for instructions: Do this task."
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 1"
        assert component == "instructions"
        assert content == "Do this task."
    
    def test_parse_complex_component_name(self):
        """Test parsing proposal with complex component name."""
        logger = RichConsoleLogger()
        message = "Iteration 5: Proposed new text for signature:EmailInput:instructions: Provide the full email."
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 5"
        assert component == "signature:EmailInput:instructions"
        assert content == "Provide the full email."
    
    def test_parse_tool_description(self):
        """Test parsing proposal for tool description."""
        logger = RichConsoleLogger()
        message = "Iteration 2: Proposed new text for tool:final_result:description: Analyze the input."
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 2"
        assert component == "tool:final_result:description"
        assert content == "Analyze the input."
    
    def test_parse_multiline_content(self):
        """Test parsing proposal with multiline content."""
        logger = RichConsoleLogger()
        content_text = """Task: Classify emails.

Instructions:
- Read the input
- Analyze sentiment
- Return results"""
        message = f"Iteration 3: Proposed new text for instructions: {content_text}"
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 3"
        assert component == "instructions"
        assert content == content_text
    
    def test_parse_non_proposal_message(self):
        """Test that non-proposal messages return None."""
        logger = RichConsoleLogger()
        
        # Test various non-proposal messages
        messages = [
            "Starting GEPA optimization...",
            "Iteration 1: Selected program 0 score: 0.713",
            "Dataset: 10 training, 10 validation examples",
            "Iteration 5: New subsample score 2.1 is better",
            "Proposed new text without iteration prefix",
        ]
        
        for message in messages:
            result = logger._parse_proposal_message(message)
            assert result is None, f"Message '{message}' should not be parsed as proposal"
    
    def test_parse_double_digit_iteration(self):
        """Test parsing proposal with double-digit iteration number."""
        logger = RichConsoleLogger()
        message = "Iteration 15: Proposed new text for instructions: Complex task."
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 15"
        assert component == "instructions"
        assert content == "Complex task."
    
    def test_parse_param_proposal(self):
        """Test parsing proposal for parameter description."""
        logger = RichConsoleLogger()
        message = "Iteration 7: Proposed new text for tool:final_result:param:urgency: Urgency level. Values: low, medium, high."
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 7"
        assert component == "tool:final_result:param:urgency"
        assert content == "Urgency level. Values: low, medium, high."
    
    def test_log_normal_message(self, capsys):
        """Test logging a normal (non-proposal) message."""
        logger = RichConsoleLogger()
        logger.log("This is a normal message")
        # Just verify it doesn't crash - Rich output isn't captured by capsys
    
    def test_log_proposal_message(self, capsys):
        """Test logging a proposal message."""
        logger = RichConsoleLogger()
        logger.log("Iteration 1: Proposed new text for instructions: Test instruction.")
        # Just verify it doesn't crash - Rich output isn't captured by capsys
    
    def test_edge_case_empty_content(self):
        """Test parsing proposal with empty content."""
        logger = RichConsoleLogger()
        message = "Iteration 1: Proposed new text for instructions: "
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 1"
        assert component == "instructions"
        assert content == ""
    
    def test_edge_case_colon_in_content(self):
        """Test parsing proposal where content contains colons."""
        logger = RichConsoleLogger()
        message = "Iteration 1: Proposed new text for instructions: Format: name: value, type: string"
        
        result = logger._parse_proposal_message(message)
        assert result is not None
        iteration, component, content = result
        
        assert iteration == "Iteration 1"
        assert component == "instructions"
        assert content == "Format: name: value, type: string"
    
    def test_integration_mixed_messages(self):
        """Test logging a mix of normal and proposal messages."""
        logger = RichConsoleLogger()
        
        messages = [
            "Dataset: 10 training examples",
            "Iteration 0: Base score: 0.7",
            "Iteration 1: Proposed new text for instructions: Classify emails by urgency.",
            "Iteration 1: Selected program score: 0.75",
            "Iteration 2: Proposed new text for signature:Input:instructions: Provide email text.",
            "Optimization complete!",
        ]
        
        # Should not raise any exceptions
        for message in messages:
            logger.log(message)
