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
import json
import sys
from pathlib import Path

import litellm
import requests
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
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
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
            completion_kwargs = {
                "model": f"openai/{self.model}",
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "api_base": f"{self.base_url}/v1",
                "api_key": "dummy-key",  # No support for API keys yet tbh
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.top_k is not None:
                completion_kwargs["top_k"] = self.top_k
            response = litellm.completion(**completion_kwargs)

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
            batch_kwargs = {
                "max_tokens": self.max_tokens,
                "api_base": f"{self.base_url}/v1",
                "api_key": "dummy-key",
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.top_k is not None:
                batch_kwargs["top_k"] = self.top_k
            responses = self.generator.api_generate(
                prompts=[self.messages.copy()],  # Can do longer context if pass OpenAI format
                model=f"openai/{self.model}",
                batch_size=1,
                batch_completion_kwargs=batch_kwargs,
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

    def change_model(self, config_path_str: str) -> bool:
        """Change the server's model by POSTing a config JSON file."""
        from sae_scoping.servers.model_configs.name_resolution import resolve_config_path

        try:
            path = resolve_config_path(config_path_str)
        except FileNotFoundError as e:
            print(f"\n\033[1;31m[Error] {e}\033[0m\n")
            return False

        try:
            with open(path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n\033[1;31m[Error] Invalid JSON in config file: {e}\033[0m\n")
            return False

        try:
            resp = requests.post(f"{self.base_url}/v1/model/change", json=config, timeout=300)
            data = resp.json()
            if data.get("success"):
                print(f"\n\033[1;32m[Success] {data.get('message', 'Model changed')}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False


def print_banner():
    """Print welcome banner."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║       HuggingFace OpenAI CLI Client (powered by LiteLLM)      ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  Commands:                                                     ║")
    print("║    /clear              - Clear conversation history            ║")
    print("║    /history            - Show conversation history             ║")
    print("║    /tokens N           - Set max tokens to N                   ║")
    print("║    /temperature F      - Set temperature (0.0 = greedy)        ║")
    print("║    /top_p F            - Set top_p for nucleus sampling        ║")
    print("║    /top_k N            - Set top_k for top-k sampling          ║")
    print("║    /change_model PATH  - Change model via config JSON file     ║")
    print("║    /help               - Show this help message                ║")
    print("║    Ctrl+C              - Exit                                  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()


def print_help():
    """Print help message."""
    print()
    print("Available commands:")
    print("  /clear              - Clear the conversation history and start fresh")
    print("  /history            - Display the current conversation history")
    print("  /tokens N           - Set max tokens to N (e.g., /tokens 1024)")
    print("  /temperature F      - Set temperature (e.g., /temperature 0.7)")
    print("                        Use 0.0 for greedy decoding")
    print("  /top_p F            - Set top_p for nucleus sampling (e.g., /top_p 0.9)")
    print("  /top_k N            - Set top_k for top-k sampling (e.g., /top_k 50)")
    print("  /change_model PATH  - Change server model using config JSON file")
    print("  /help               - Show this help message")
    print("  Ctrl+C              - Exit the client")
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
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, use 0.0 for greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter (default: None)",
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
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
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
                    elif cmd.startswith("/tokens"):
                        parts = user_input.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            n = int(parts[1])
                            if n >= 0:
                                client.max_tokens = n
                                print(f"\n\033[1;33m[Max tokens set to {client.max_tokens}]\033[0m\n")
                            else:
                                print("\n\033[1;31mUsage: /tokens N (e.g., /tokens 1024) where N >= 0\033[0m\n")
                        else:
                            print("\n\033[1;31mUsage: /tokens N (e.g., /tokens 1024)\033[0m\n")
                        continue
                    elif cmd.startswith("/temperature"):
                        parts = user_input.split()
                        if len(parts) == 2:
                            try:
                                t = float(parts[1])
                                if t >= 0:
                                    client.temperature = t
                                    mode = "greedy" if t == 0.0 else f"sampling (temp={t})"
                                    print(f"\n\033[1;33m[Temperature set to {t} ({mode})]\033[0m\n")
                                else:
                                    print("\n\033[1;31mTemperature must be >= 0\033[0m\n")
                            except ValueError:
                                print("\n\033[1;31mUsage: /temperature F (e.g., /temperature 0.7)\033[0m\n")
                        else:
                            print("\n\033[1;31mUsage: /temperature F (e.g., /temperature 0.7)\033[0m\n")
                        continue
                    elif cmd.startswith("/top_p"):
                        parts = user_input.split()
                        if len(parts) == 2:
                            try:
                                p = float(parts[1])
                                if 0 < p <= 1:
                                    client.top_p = p
                                    print(f"\n\033[1;33m[top_p set to {p}]\033[0m\n")
                                else:
                                    print("\n\033[1;31mtop_p must be in (0, 1]\033[0m\n")
                            except ValueError:
                                print("\n\033[1;31mUsage: /top_p F (e.g., /top_p 0.9)\033[0m\n")
                        else:
                            print("\n\033[1;31mUsage: /top_p F (e.g., /top_p 0.9)\033[0m\n")
                        continue
                    elif cmd.startswith("/top_k"):
                        parts = user_input.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            k = int(parts[1])
                            if k > 0:
                                client.top_k = k
                                print(f"\n\033[1;33m[top_k set to {k}]\033[0m\n")
                            else:
                                print("\n\033[1;31mtop_k must be > 0\033[0m\n")
                        else:
                            print("\n\033[1;31mUsage: /top_k N (e.g., /top_k 50)\033[0m\n")
                        continue
                    elif cmd.startswith("/change_model"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_model(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /change_model PATH (e.g., /change_model config.json)\033[0m\n")
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
