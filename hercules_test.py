#!/usr/bin/env python3
"""TUI script to test Hercules GPU server connection and functionality.

Run with: python hercules_test.py
"""

from __future__ import annotations

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header():
    """Print the TUI header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          Hercules GPU Server Connection Tester             ║")
    print("║                  WhisperSSTis v1.0                         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")


def print_status(label: str, status: str, ok: bool):
    """Print a status line."""
    color = Colors.GREEN if ok else Colors.RED
    symbol = "✓" if ok else "✗"
    print(f"  {label:<30} [{color}{symbol}{Colors.END}] {status}")


def print_info(label: str, value: str):
    """Print an info line."""
    print(f"  {Colors.DIM}{label:<30}{Colors.END} {value}")


def get_hercules_url() -> str:
    """Get Hercules URL from environment or default."""
    return os.getenv("HERCULES_URL", "http://172.26.12.7:8000")


def test_network_connectivity():
    """Test basic network connectivity to Hercules."""
    print_section("Network Connectivity")

    import subprocess

    host = "172.26.12.7"

    # Ping test
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-t", "2", host],
            capture_output=True,
            timeout=5
        )
        ping_ok = result.returncode == 0
        if ping_ok:
            # Extract latency
            output = result.stdout.decode()
            if "time=" in output:
                latency = output.split("time=")[1].split()[0]
                print_status("Ping", f"{latency}", True)
            else:
                print_status("Ping", "reachable", True)
        else:
            print_status("Ping", "unreachable", False)
    except Exception as e:
        print_status("Ping", f"error: {e}", False)
        ping_ok = False

    return ping_ok


def test_http_connection():
    """Test HTTP connection to Hercules server."""
    print_section("HTTP Connection")

    url = get_hercules_url()
    print_info("Server URL", url)

    try:
        import requests

        start = time.time()
        response = requests.get(f"{url}/health", timeout=10)
        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            print_status("HTTP GET /health", f"{elapsed:.0f}ms", True)
            return True, response.json()
        else:
            print_status("HTTP GET /health", f"status {response.status_code}", False)
            return False, None
    except requests.exceptions.ConnectionError:
        print_status("HTTP GET /health", "connection refused", False)
        print(f"\n  {Colors.YELLOW}Hint: Is the server running on Hercules?{Colors.END}")
        print(f"  {Colors.DIM}Start with: uvicorn hercules_server:app --host 0.0.0.0 --port 8000{Colors.END}")
        return False, None
    except requests.exceptions.Timeout:
        print_status("HTTP GET /health", "timeout", False)
        return False, None
    except Exception as e:
        print_status("HTTP GET /health", f"error: {e}", False)
        return False, None


def display_server_info(health: dict):
    """Display server health information."""
    print_section("Server Status")

    print_status("Server", health.get("status", "unknown"), health.get("status") == "ok")
    print_status("CUDA Available", str(health.get("cuda_available", False)), health.get("cuda_available", False))

    # GPUs
    gpus = health.get("gpus", [])
    if gpus:
        print(f"\n  {Colors.BOLD}GPUs:{Colors.END}")
        for i, gpu in enumerate(gpus):
            print(f"    {Colors.GREEN}[{i}]{Colors.END} {gpu}")
    else:
        print(f"\n  {Colors.YELLOW}No GPUs detected{Colors.END}")

    # Loaded models
    models = health.get("loaded_models", [])
    if models:
        print(f"\n  {Colors.BOLD}Loaded Whisper Models:{Colors.END}")
        for model in models:
            print(f"    {Colors.CYAN}•{Colors.END} {model}")
    else:
        print(f"\n  {Colors.DIM}No models loaded yet (will load on first request){Colors.END}")

    # Ollama
    ollama_ok = health.get("ollama_available", False)
    print_status("\nOllama", "available" if ollama_ok else "not available", ollama_ok)


def test_transcription():
    """Test transcription endpoint with a simple audio sample."""
    print_section("Transcription Test")

    try:
        import numpy as np
        import requests
        import io
        import soundfile as sf
    except ImportError as e:
        print_status("Dependencies", f"missing: {e}", False)
        return False

    url = get_hercules_url()

    # Generate a short silent audio sample (1 second)
    print_info("Generating", "1 second test audio")
    sample_rate = 16000
    duration = 1.0
    audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Add a tiny bit of noise so it's not completely silent
    audio_data += np.random.randn(len(audio_data)).astype(np.float32) * 0.001

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    # Send to server
    print_info("Sending to", f"{url}/transcribe")

    try:
        start = time.time()
        response = requests.post(
            f"{url}/transcribe",
            files={"file": ("test.wav", buffer, "audio/wav")},
            data={"model_key": "icelandic"},
            timeout=120
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print_status("Transcription", f"{elapsed:.1f}s", True)
            print_info("Model used", result.get("model", "unknown"))
            print_info("Device", result.get("device", "unknown"))
            print_info("Result", f'"{result.get("text", "")}"' or "(silence)")
            return True
        else:
            print_status("Transcription", f"status {response.status_code}", False)
            try:
                error = response.json().get("detail", response.text)
                print(f"  {Colors.RED}Error: {error}{Colors.END}")
            except:
                pass
            return False
    except requests.exceptions.Timeout:
        print_status("Transcription", "timeout (model may be loading)", False)
        print(f"  {Colors.YELLOW}Hint: First request loads the model, try again{Colors.END}")
        return False
    except Exception as e:
        print_status("Transcription", f"error: {e}", False)
        return False


def test_ollama():
    """Test Ollama LLM endpoint."""
    print_section("Ollama LLM Test")

    try:
        import requests
    except ImportError:
        print_status("Dependencies", "requests not installed", False)
        return False

    url = get_hercules_url()

    # First check available models
    print_info("Checking", "available models")

    try:
        response = requests.get(f"{url}/ollama/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if models:
                print(f"  {Colors.BOLD}Available models:{Colors.END}")
                for m in models[:5]:  # Show first 5
                    print(f"    {Colors.CYAN}•{Colors.END} {m.get('name', m)}")
            else:
                print(f"  {Colors.YELLOW}No models found. Run: ollama pull llama3{Colors.END}")
                return False
        else:
            print_status("List models", f"status {response.status_code}", False)
            return False
    except Exception as e:
        print_status("List models", f"error: {e}", False)
        return False

    # Test a simple completion
    print_info("\nTesting", "LLM completion")

    payload = {
        "prompt": "Say 'hello' in Icelandic. Reply with just the word.",
        "model": "llama3",
        "temperature": 0.1,
        "max_tokens": 50
    }

    try:
        start = time.time()
        response = requests.post(f"{url}/llm", json=payload, timeout=60)
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            print_status("LLM Completion", f"{elapsed:.1f}s", True)
            print_info("Model", result.get("model", "unknown"))
            print_info("Response", f'"{result.get("text", "")[:100]}"')
            return True
        else:
            print_status("LLM Completion", f"status {response.status_code}", False)
            try:
                error = response.json().get("detail", "")
                print(f"  {Colors.RED}Error: {error}{Colors.END}")
            except:
                pass
            return False
    except requests.exceptions.Timeout:
        print_status("LLM Completion", "timeout", False)
        return False
    except Exception as e:
        print_status("LLM Completion", f"error: {e}", False)
        return False


def print_summary(results: dict):
    """Print test summary."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                        Summary                             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, ok in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if ok else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {test:<30} {status}")

    print(f"\n  {Colors.BOLD}Result: {passed}/{total} tests passed{Colors.END}")

    if passed == total:
        print(f"\n  {Colors.GREEN}✓ Hercules is ready for transcription!{Colors.END}")
    elif passed > 0:
        print(f"\n  {Colors.YELLOW}⚠ Some features may not work{Colors.END}")
    else:
        print(f"\n  {Colors.RED}✗ Cannot connect to Hercules{Colors.END}")
        print(f"\n  {Colors.DIM}Check:{Colors.END}")
        print(f"  {Colors.DIM}1. VPN is connected{Colors.END}")
        print(f"  {Colors.DIM}2. Server is running on Hercules{Colors.END}")
        print(f"  {Colors.DIM}3. Firewall allows port 8000{Colors.END}")


def show_menu():
    """Show interactive menu."""
    print(f"\n{Colors.BOLD}Options:{Colors.END}")
    print("  [1] Run all tests")
    print("  [2] Test network only")
    print("  [3] Test HTTP connection")
    print("  [4] Test transcription")
    print("  [5] Test Ollama LLM")
    print("  [q] Quit")
    print()
    return input(f"{Colors.CYAN}Select option: {Colors.END}").strip().lower()


def main():
    """Main TUI entry point."""
    print_header()

    # Check for required dependencies
    try:
        import requests
    except ImportError:
        print(f"{Colors.RED}Error: 'requests' package required{Colors.END}")
        print(f"Install with: pip install requests")
        sys.exit(1)

    while True:
        choice = show_menu()

        if choice == 'q':
            print(f"\n{Colors.DIM}Goodbye!{Colors.END}\n")
            break
        elif choice == '1':
            # Run all tests
            results = {}
            results["Network"] = test_network_connectivity()
            http_ok, health = test_http_connection()
            results["HTTP"] = http_ok
            if http_ok and health:
                display_server_info(health)
                results["Transcription"] = test_transcription()
                results["Ollama LLM"] = test_ollama()
            print_summary(results)
        elif choice == '2':
            test_network_connectivity()
        elif choice == '3':
            http_ok, health = test_http_connection()
            if http_ok and health:
                display_server_info(health)
        elif choice == '4':
            test_transcription()
        elif choice == '5':
            test_ollama()
        else:
            print(f"{Colors.YELLOW}Invalid option{Colors.END}")


if __name__ == "__main__":
    main()
