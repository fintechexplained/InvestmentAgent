"""Pytest configuration for all tests."""

import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def mock_api_keys():
    """Set fake API keys for all tests.

    This fixture automatically runs before any tests and ensures that
    API clients can be initialized without requiring real API keys.

    The clients are mocked in individual tests, but they need valid-looking
    keys during initialization to avoid exceptions.
    """
    # Store original values (if they exist)
    original_openai = os.environ.get("OPENAI_API_KEY")
    original_anthropic = os.environ.get("ANTHROPIC_API_KEY")

    # Set fake keys that look valid but won't work with real APIs
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-unit-testing-only-1234567890"
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-fake-key-for-unit-testing-only-1234567890"

    yield  # All tests run with these fake keys

    # Restore original values after all tests complete
    if original_openai:
        os.environ["OPENAI_API_KEY"] = original_openai
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    if original_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = original_anthropic
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)
