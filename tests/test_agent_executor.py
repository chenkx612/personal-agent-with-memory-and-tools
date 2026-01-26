import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import get_agent_executor

class TestAgentExecutor(unittest.TestCase):
    def test_get_agent_executor_initialization(self):
        # Mock environment variable
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"}):
            # Mock ChatOpenAI to avoid actual network calls
            with patch("src.agent.ChatOpenAI") as MockChatOpenAI:
                # Mock the instance
                mock_llm = MagicMock()
                MockChatOpenAI.return_value = mock_llm
                
                # Call the function
                try:
                    agent = get_agent_executor()
                    print("Agent initialized successfully")
                    self.assertIsNotNone(agent)
                except Exception as e:
                    self.fail(f"Agent initialization failed with Exception: {e}")

if __name__ == "__main__":
    unittest.main()
