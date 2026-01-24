#!/usr/bin/env python3
"""
Interactive CLI client for HuggingFace OpenAI-compatible API server.
Uses LiteLLM to communicate with the server.

Usage:
    python -m sae_scoping.servers.hf_openai_cli_client
    python -m sae_scoping.servers.hf_openai_cli_client --base-url http://localhost:8080
    python -m sae_scoping.servers.hf_openai_cli_client --model "custom-model-name"

Implement by Claude.
"""

from __future__ import annotations
import argparse
import sys
import os

import litellm
from sae_scoping.utils.generation.api_generator import APIGenerator


# Disable LiteLLM verbose logging
litellm.set_verbose = False


class InteractiveChatClient:
    """Interactive chat client using LiteLLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "huggingface/default",
        system_prompt: str | None = None,
        max_tokens: int = 512,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.generator = APIGenerator()

        # Conversation history
        self.messages: list[dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send_message(self, user_input: str) -> str | None:
        """Send a message and get a response."""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        try:
            # Use LiteLLM directly for single message
            # The openai/ prefix tells LiteLLM to use OpenAI-compatible API
            response = litellm.completion(
                model=f"openai/{self.model}",
                messages=self.messages,
                max_tokens=self.max_tokens,
                api_base=f"{self.base_url}/v1",
                api_key="dummy-key",  # No support for API keys yet tbh
            )

            # Extract response content
            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": assistant_message})

            return assistant_message

        except Exception as e:
            # Remove the failed user message from history
            self.messages.pop()
            return f"[Error] {type(e).__name__}: {e}"

    def send_message_with_generator(self, user_input: str) -> str | None:
        """Send a message using APIGenerator (for batch compatibility testing)."""
        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        try:
            # Use APIGenerator
            responses = self.generator.api_generate(
                prompts=[self.messages.copy()],  # Can do longer context if pass OpenAI format
                model=f"openai/{self.model}",
                batch_size=1,
                batch_completion_kwargs={
                    "max_tokens": self.max_tokens,
                    "api_base": f"{self.base_url}/v1",
                    "api_key": "dummy-key",
                },
            )

            if responses and responses[0] is not None:
                assistant_message = responses[0]
                self.messages.append({"role": "assistant", "content": assistant_message})
                return assistant_message
            else:
                self.messages.pop()
                return "[Error] No response received"

        except Exception as e:
            self.messages.pop()
            return f"[Error] {type(e).__name__}: {e}"

    def clear_history(self):
        """Clear conversation history (keeping system prompt if any)."""
        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages

    def print_history(self):
        """Print the current conversation history."""
        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)
        for i, msg in enumerate(self.messages):
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate long messages for display
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"[{i}] {role}: {content}")
        print("=" * 60 + "\n")


def print_banner():
    """Print welcome banner."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     HuggingFace OpenAI CLI Client (powered by LiteLLM)    ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║  Commands:                                                 ║")
    print("║    /clear  - Clear conversation history                    ║")
    print("║    /history - Show conversation history                    ║")
    print("║    /help   - Show this help message                        ║")
    print("║    Ctrl+C  - Exit                                          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def print_help():
    """Print help message."""
    print()
    print("Available commands:")
    print("  /clear   - Clear the conversation history and start fresh")
    print("  /history - Display the current conversation history")
    print("  /help    - Show this help message")
    print("  Ctrl+C   - Exit the client")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI client for HuggingFace OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect to default server
    python -m sae_scoping.servers.hf_openai_cli_client
    
    # Connect to custom server
    python -m sae_scoping.servers.hf_openai_cli_client --base-url http://localhost:8080
    
    # Use with system prompt
    python -m sae_scoping.servers.hf_openai_cli_client --system "You are a helpful math tutor."
    
    # Use APIGenerator mode (for testing batch compatibility)
    python -m sae_scoping.servers.hf_openai_cli_client --use-generator
        """,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the OpenAI-compatible server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name to use (default: default)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt to set context",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in response (default: 512)",
    )
    parser.add_argument(
        "--use-generator",
        action="store_true",
        help="Use APIGenerator instead of direct LiteLLM (for batch testing)",
    )

    args = parser.parse_args()

    # Create client
    client = InteractiveChatClient(
        base_url=args.base_url,
        model=args.model,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
    )

    # Choose send method
    send_fn = client.send_message_with_generator if args.use_generator else client.send_message

    # Print banner
    print_banner()
    print(f"Connected to: {args.base_url}")
    print(f"Model: {args.model}")
    if args.system:
        print(f"System prompt: {args.system[:50]}{'...' if len(args.system) > 50 else ''}")
    if args.use_generator:
        print("Mode: APIGenerator (batch testing)")
    print()

    # Main loop
    try:
        while True:
            try:
                # Get user input
                user_input = input("\033[1;32mYou:\033[0m ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd == "/clear":
                        client.clear_history()
                        print("\n\033[1;33m[Conversation cleared]\033[0m\n")
                        continue
                    elif cmd == "/history":
                        client.print_history()
                        continue
                    elif cmd == "/help":
                        print_help()
                        continue
                    else:
                        print(f"\n\033[1;31mUnknown command: {user_input}\033[0m")
                        print("Type /help for available commands.\n")
                        continue

                # Send message and get response
                print("\033[1;34mAssistant:\033[0m ", end="", flush=True)
                response = send_fn(user_input)
                print(response)
                print()

            except EOFError:
                # Handle Ctrl+D
                print("\n")
                break

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n\033[1;33mGoodbye!\033[0m\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
