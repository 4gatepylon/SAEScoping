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

    def get_server_config(self) -> dict | None:
        """Fetch current server configuration."""
        try:
            resp = requests.get(f"{self.base_url}/v1/model/config", timeout=30)
            return resp.json()
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Failed to fetch config: {e}\033[0m\n")
            return None

    def print_sae_mode(self) -> None:
        """Print current SAE mode (sparsify or saelens)."""
        data = self.get_server_config()
        if data is None:
            return

        config = data.get("config")
        if config is None:
            print("\n\033[1;33m[Info] No model configuration loaded\033[0m\n")
            return

        sae_mode = config.get("sae_mode")
        sae_path = config.get("sae_path")
        sae_release = config.get("sae_release")
        sae_id = config.get("sae_id")
        distribution_path = config.get("distribution_path")

        print("\n" + "=" * 50)
        print("SAE CONFIGURATION")
        print("=" * 50)
        print(f"  Mode: {sae_mode or 'none'}")
        if sae_mode == "sparsify" or sae_path:
            print(f"  SAE Path: {sae_path or 'not set'}")
        if sae_mode == "saelens" or sae_release:
            print(f"  SAE Release: {sae_release or 'not set'}")
            print(f"  SAE ID: {sae_id or 'not set'}")
            print(f"  Distribution Path: {distribution_path or 'not set'}")
        print("=" * 50 + "\n")

    def print_config(self) -> None:
        """Print full model configuration in a nicely formatted way."""
        data = self.get_server_config()
        if data is None:
            return

        config = data.get("config")
        model_name = data.get("model", "unknown")

        if config is None:
            print("\n\033[1;33m[Info] No model configuration loaded\033[0m\n")
            return

        print("\n" + "=" * 60)
        print("MODEL CONFIGURATION")
        print("=" * 60)

        # Model info
        print("\n\033[1;36m[Model]\033[0m")
        print(f"  model_name_or_path: {config.get('model_name_or_path', 'N/A')}")
        print(f"  loaded_as: {model_name}")

        # Attention config
        attn = config.get("attn_implementation")
        if attn:
            print(f"  attn_implementation: {attn}")

        # SAE config
        sae_mode = config.get("sae_mode")
        sae_path = config.get("sae_path")
        sae_release = config.get("sae_release")
        if sae_mode or sae_path or sae_release:
            print("\n\033[1;36m[SAE]\033[0m")
            print(f"  sae_mode: {sae_mode or 'auto'}")
            if sae_path:
                print(f"  sae_path: {sae_path}")
            if sae_release:
                print(f"  sae_release: {sae_release}")
                print(f"  sae_id: {config.get('sae_id', 'N/A')}")
            hookpoint = config.get("hookpoint")
            if hookpoint:
                print(f"  hookpoint: {hookpoint}")

        # Pruning config
        dist_path = config.get("distribution_path")
        prune_threshold = config.get("prune_threshold")
        if dist_path or prune_threshold:
            print("\n\033[1;36m[Pruning]\033[0m")
            print(f"  distribution_path: {dist_path or 'N/A'}")
            print(f"  prune_threshold: {prune_threshold if prune_threshold is not None else 'N/A'}")

        # Server settings
        print("\n\033[1;36m[Server]\033[0m")
        print(f"  batch_size: {config.get('batch_size', 1)}")
        print(f"  sleep_time: {config.get('sleep_time', 0.0)}")
        chat_template_path = config.get("chat_template_path")
        if chat_template_path:
            print(f"  chat_template_path: {chat_template_path}")
        if config.get("test_mode"):
            print("  test_mode: True")

        print("\n" + "=" * 60 + "\n")

    def change_sae_path(self, sae_input: str) -> bool:
        """Change Sparsify SAE path (only valid when in sparsify mode)."""
        from sae_scoping.servers.model_configs.name_resolution import (
            resolve_sae_artifact_path,
            validate_sparsify_sae_dir,
            DEFAULT_SPARSIFY_SAE_OUTPUT_DIR,
        )

        # Fetch current config
        data = self.get_server_config()
        if data is None:
            return False

        config = data.get("config")
        if config is None:
            print("\n\033[1;31m[Error] No model configuration loaded\033[0m\n")
            return False

        # Validate we're in sparsify mode
        sae_mode = config.get("sae_mode")
        if sae_mode != "sparsify" and config.get("sae_path") is None:
            print(f"\n\033[1;31m[Error] Server is not in sparsify mode (current mode: {sae_mode})\033[0m")
            print("Use /change_distribution for SAELens mode.\n")
            return False

        # Resolve and validate path client-side
        try:
            resolved_path = resolve_sae_artifact_path(
                sae_input,
                file_glob="sae.safetensors",
                default_dir=DEFAULT_SPARSIFY_SAE_OUTPUT_DIR,
                validator=validate_sparsify_sae_dir,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"\n\033[1;31m[Error] {e}\033[0m\n")
            return False

        # Build change request with only the diff
        change_config = config.copy()
        change_config["sae_path"] = str(resolved_path)

        try:
            resp = requests.post(f"{self.base_url}/v1/model/change", json=change_config, timeout=300)
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] SAE changed to: {resolved_path}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False

    def change_distribution_path(self, dist_input: str) -> bool:
        """Change SAELens distribution path (only valid when in saelens mode)."""
        # Fetch current config
        data = self.get_server_config()
        if data is None:
            return False

        config = data.get("config")
        if config is None:
            print("\n\033[1;31m[Error] No model configuration loaded\033[0m\n")
            return False

        # Validate we're in saelens mode
        sae_mode = config.get("sae_mode")
        if sae_mode != "saelens" and config.get("sae_release") is None:
            print(f"\n\033[1;31m[Error] Server is not in saelens mode (current mode: {sae_mode})\033[0m")
            print("Use /change_sae for Sparsify mode.\n")
            return False

        # Validate path exists client-side
        dist_path = Path(dist_input)
        if not dist_path.exists():
            print(f"\n\033[1;31m[Error] Distribution file not found: {dist_input}\033[0m\n")
            return False
        if not dist_path.name.endswith(".safetensors"):
            print(f"\n\033[1;31m[Error] Distribution file must be a .safetensors file: {dist_path.name}\033[0m\n")
            return False

        # Build change request with only the diff
        change_config = config.copy()
        change_config["distribution_path"] = str(dist_path.resolve())

        try:
            resp = requests.post(f"{self.base_url}/v1/model/change", json=change_config, timeout=300)
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] Distribution changed to: {dist_path}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False

    def change_model_only(self, model_name_or_path: str) -> bool:
        """Change only the model, keeping SAE and other config intact."""
        # Fetch current config
        data = self.get_server_config()
        if data is None:
            return False

        config = data.get("config")
        if config is None:
            print("\n\033[1;31m[Error] No model configuration loaded\033[0m\n")
            return False

        # Build change request with only model_name_or_path changed
        change_config = config.copy()
        change_config["model_name_or_path"] = model_name_or_path

        try:
            resp = requests.post(f"{self.base_url}/v1/model/change", json=change_config, timeout=300)
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] Model changed to: {model_name_or_path}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False

    def change_batch_size(self, batch_size: int) -> bool:
        """Change server batch size without model reload."""
        if batch_size < 1:
            print("\n\033[1;31m[Error] batch_size must be >= 1\033[0m\n")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/v1/settings",
                json={"batch_size": batch_size},
                timeout=30,
            )
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] {resp_data.get('message')}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False

    def change_sleep_time(self, sleep_time: float) -> bool:
        """Change server sleep time without model reload."""
        if sleep_time < 0:
            print("\n\033[1;31m[Error] sleep_time must be >= 0\033[0m\n")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/v1/settings",
                json={"sleep_time": sleep_time},
                timeout=30,
            )
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] {resp_data.get('message')}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False

    def change_chat_template(self, template_path: str) -> bool:
        """Change server chat template without model reload."""
        # Validate path exists client-side
        path = Path(template_path)
        if not path.exists():
            print(f"\n\033[1;31m[Error] Chat template file not found: {template_path}\033[0m\n")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/v1/settings",
                json={"chat_template_path": str(path.resolve())},
                timeout=30,
            )
            resp_data = resp.json()
            if resp_data.get("success"):
                print(f"\n\033[1;32m[Success] {resp_data.get('message')}\033[0m\n")
                return True
            else:
                print(f"\n\033[1;31m[Failed] {resp_data.get('message', 'Unknown error')}\033[0m\n")
                return False
        except requests.RequestException as e:
            print(f"\n\033[1;31m[Error] Request failed: {e}\033[0m\n")
            return False


def print_banner():
    """Print welcome banner."""
    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║        HuggingFace OpenAI CLI Client (powered by LiteLLM)        ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║  Commands:                                                        ║")
    print("║    /clear                - Clear conversation history             ║")
    print("║    /history              - Show conversation history              ║")
    print("║    /tokens N             - Set max tokens to N                    ║")
    print("║    /temperature F        - Set temperature (0.0 = greedy)         ║")
    print("║    /top_p F              - Set top_p for nucleus sampling         ║")
    print("║    /top_k N              - Set top_k for top-k sampling           ║")
    print("║    /config               - Show full model configuration          ║")
    print("║    /change_config PATH   - Change model via config JSON file      ║")
    print("║    /change_model MODEL   - Change only the model name/path        ║")
    print("║    /change_sae PATH      - Change Sparsify SAE path               ║")
    print("║    /change_distribution  - Change SAELens distribution path       ║")
    print("║    /sae_mode             - Show current SAE configuration         ║")
    print("║    /batch_size N         - Set server batch size                  ║")
    print("║    /sleep_time F         - Set server batch sleep time            ║")
    print("║    /chat_template PATH   - Set chat template from file            ║")
    print("║    /help                 - Show this help message                 ║")
    print("║    Ctrl+C                - Exit                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()


def print_help():
    """Print help message."""
    print()
    print("Available commands:")
    print()
    print("  CONVERSATION:")
    print("    /clear                - Clear the conversation history and start fresh")
    print("    /history              - Display the current conversation history")
    print()
    print("  GENERATION SETTINGS (client-side):")
    print("    /tokens N             - Set max tokens to N (e.g., /tokens 1024)")
    print("    /temperature F        - Set temperature (e.g., /temperature 0.7)")
    print("                            Use 0.0 for greedy decoding")
    print("    /top_p F              - Set top_p for nucleus sampling (e.g., /top_p 0.9)")
    print("    /top_k N              - Set top_k for top-k sampling (e.g., /top_k 50)")
    print()
    print("  MODEL CONFIGURATION (server-side, requires reload):")
    print("    /config               - Show full server model configuration")
    print("    /change_config PATH   - Change full config via JSON file")
    print("    /change_model MODEL   - Change only model name/path (keeps SAE config)")
    print("    /change_sae PATH      - Change Sparsify SAE path (sparsify mode only)")
    print("    /change_distribution  - Change SAELens distribution (saelens mode only)")
    print("    /sae_mode             - Show current SAE mode and configuration")
    print()
    print("  SERVER SETTINGS (no model reload):")
    print("    /batch_size N         - Set server batch size (e.g., /batch_size 16)")
    print("    /sleep_time F         - Set batch accumulation time (e.g., /sleep_time 2.0)")
    print("    /chat_template PATH   - Load chat template from file")
    print()
    print("  OTHER:")
    print("    /help                 - Show this help message")
    print("    Ctrl+C                - Exit the client")
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
                    elif cmd.startswith("/change_config"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_model(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /change_config PATH (e.g., /change_config config.json)\033[0m\n")
                        continue
                    elif cmd.startswith("/change_model"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_model_only(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /change_model MODEL (e.g., /change_model google/gemma-2-9b-it)\033[0m\n")
                        continue
                    elif cmd.startswith("/change_sae"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_sae_path(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /change_sae PATH (e.g., /change_sae /path/to/sae)\033[0m\n")
                        continue
                    elif cmd.startswith("/change_distribution"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_distribution_path(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /change_distribution PATH (e.g., /change_distribution /path/to/dist.safetensors)\033[0m\n")
                        continue
                    elif cmd == "/sae_mode":
                        client.print_sae_mode()
                        continue
                    elif cmd == "/config":
                        client.print_config()
                        continue
                    elif cmd.startswith("/batch_size"):
                        parts = user_input.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            n = int(parts[1])
                            client.change_batch_size(n)
                        else:
                            print("\n\033[1;31mUsage: /batch_size N (e.g., /batch_size 16)\033[0m\n")
                        continue
                    elif cmd.startswith("/sleep_time"):
                        parts = user_input.split()
                        if len(parts) == 2:
                            try:
                                t = float(parts[1])
                                client.change_sleep_time(t)
                            except ValueError:
                                print("\n\033[1;31mUsage: /sleep_time F (e.g., /sleep_time 2.0)\033[0m\n")
                        else:
                            print("\n\033[1;31mUsage: /sleep_time F (e.g., /sleep_time 2.0)\033[0m\n")
                        continue
                    elif cmd.startswith("/chat_template"):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 2:
                            client.change_chat_template(parts[1].strip())
                        else:
                            print("\n\033[1;31mUsage: /chat_template PATH (e.g., /chat_template template.jinja2)\033[0m\n")
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
