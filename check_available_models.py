"""Test different Anthropic model names to find which ones work."""

import asyncio
import os
from pydantic_ai import Agent


async def test_model(model_name: str) -> bool:
    """Test if a model name works with pydantic-ai."""
    try:
        agent = Agent(model=model_name)
        result = await agent.run("Say 'ok'")
        print(f"[SUCCESS] {model_name}")
        print(f"  Response: {result.output}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not_found" in error_msg:
            print(f"[NOT FOUND] {model_name}")
        else:
            print(f"[ERROR] {model_name}")
            print(f"  Error: {error_msg[:100]}")
        return False


async def main():
    """Test various model name formats."""

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        return

    print("Testing Anthropic model names...\n")

    # Various model name formats to try
    model_names = [
        # With anthropic: prefix
        "anthropic:claude-3-5-sonnet-20241022",
        "anthropic:claude-3-5-sonnet-20240620",
        "anthropic:claude-3-opus-20240229",
        "anthropic:claude-3-sonnet-20240229",
        "anthropic:claude-3-haiku-20240307",

        # Without prefix
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",

        # Latest versions
        "claude-3-5-sonnet-latest",
        "claude-3-opus-latest",
        "claude-3-sonnet-latest",
        "claude-3-haiku-latest",
    ]

    working_models = []

    for model_name in model_names:
        success = await test_model(model_name)
        if success:
            working_models.append(model_name)
        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)

    if working_models:
        print(f"\n[SUCCESS] Working models ({len(working_models)}):")
        for model in working_models:
            print(f"  - {model}")
    else:
        print("\n[FAILED] No working models found!")
        print("\nPossible issues:")
        print("  1. API key is invalid or doesn't have access")
        print("  2. pydantic-ai version incompatibility")
        print("  3. Network/firewall issues")


if __name__ == "__main__":
    asyncio.run(main())
