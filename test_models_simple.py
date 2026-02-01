"""Simple test to check Anthropic API access and available models."""

import os
from anthropic import Anthropic

def main():
    """Test direct Anthropic API access."""

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        return

    print("Testing Anthropic API directly...\n")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}\n")

    client = Anthropic(api_key=api_key)

    # Test different model names
    model_names = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    working_models = []

    for model_name in model_names:
        try:
            print(f"Testing: {model_name}...", end=" ")
            message = client.messages.create(
                model=model_name,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "Say 'ok'"}
                ]
            )
            print(f"[SUCCESS]")
            print(f"  Response: {message.content[0].text}")
            working_models.append(model_name)
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not_found" in error_msg:
                print(f"[NOT FOUND]")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                print(f"[AUTH ERROR]")
                print(f"  Error: {error_msg}")
                break
            else:
                print(f"[ERROR]")
                print(f"  Error: {error_msg[:150]}")

    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)

    if working_models:
        print(f"\nWorking models ({len(working_models)}):")
        for model in working_models:
            print(f"  - {model}")
        print(f"\nRecommendation: Update investment_agent.py to use:")
        print(f'  model="{working_models[0]}"')
    else:
        print("\nNo working models found!")
        print("\nThis could mean:")
        print("  1. Your API key doesn't have access to Claude models")
        print("  2. Account type limitation (free tier?)")
        print("  3. Regional restrictions")
        print("  4. API key is invalid")


if __name__ == "__main__":
    main()
