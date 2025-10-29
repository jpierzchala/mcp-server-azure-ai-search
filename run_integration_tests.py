#!/usr/bin/env python3
"""
Test runner script for Azure AI Search MCP Integration Tests.

This script checks if integration tests are enabled and runs them
with proper configuration and safety prompts.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    print("Azure AI Search MCP Integration Test Runner")
    print("=" * 40)

    integration_enabled = os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"
    if not integration_enabled:
        print("❌ Integration tests are DISABLED")
        print()
        print("To enable integration tests:")
        print("1. Create .env and set ENABLE_INTEGRATION_TESTS=true")
        print("2. Configure the following variables:")
        print("   - AZURE_SEARCH_SERVICE_ENDPOINT (e.g., https://<service>.search.windows.net)")
        print("   - AZURE_SEARCH_API_KEY")
        print("   - AZURE_SEARCH_INDEX_NAME")
        return 1

    # Check required configuration
    required_vars = [
        "AZURE_SEARCH_SERVICE_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        print("Please configure these variables in your .env file")
        return 1

    print("✅ Integration tests are ENABLED")
    print("✅ Required configuration found")

    print()
    print("Test Configuration:")
    print(f"  Endpoint: {os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')}")
    print(f"  Index: {os.getenv('AZURE_SEARCH_INDEX_NAME')}")

    print()
    print("⚠️  WARNING: These tests will connect to a real Azure AI Search index")
    print("   and may perform search queries. Continue? (y/N): ", end="")

    if "--yes" in sys.argv or "-y" in sys.argv:
        response = "y"
        print("y (auto-confirmed)")
    else:
        response = input().lower()

    if response != "y":
        print("❌ Tests cancelled by user")
        return 1

    print()
    print("🚀 Running integration tests...")
    print("=" * 40)

    test_file = Path(__file__).parent / "tests" / "integration" / "test_integration.py"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--tb=short",
        "--color=yes",
    ]

    # Pass through additional pytest arguments
    passthrough = [arg for arg in sys.argv[1:] if arg not in ["--yes", "-y"]]
    cmd.extend(passthrough)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


