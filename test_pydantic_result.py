"""Test to see the structure of pydantic-ai result."""

import asyncio
from pydantic_ai import Agent


async def test_result_structure():
    """Check what attributes the result has."""
    agent = Agent(model="anthropic:claude-3-haiku-20240307")

    result = await agent.run("Say hello")

    print("Result type:", type(result))
    print("\nResult attributes:")
    for attr in dir(result):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    print("\nTrying different ways to access the response:")

    # Try different attributes
    attrs_to_try = ['data', 'output', 'text', 'content', 'response', 'message']

    for attr in attrs_to_try:
        if hasattr(result, attr):
            value = getattr(result, attr)
            print(f"  result.{attr} = {value} (type: {type(value).__name__})")

    print(f"\nDirect result: {result}")
    print(f"Result repr: {repr(result)}")


if __name__ == "__main__":
    asyncio.run(test_result_structure())
