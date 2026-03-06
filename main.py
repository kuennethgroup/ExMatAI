"""
ExMat AI - Main Entry Point
Automated Battery Material Data Extraction Pipeline
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

# Load environment
load_dotenv()

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/exmatai.log"),
        logging.StreamHandler(),
    ],
)


def print_banner():
    """Print application banner."""
    banner = f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════════╗
║                          ExMat AI                              ║
║        Automated Battery Material Data Extraction              ║
║                                                                ║
║  Powered by: DeepSeek-OCR • Qwen3.5 • Qwen3-VL • MolDetV2      ║
╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def check_environment() -> bool:
    """Check if the environment is properly set up."""
    print(f"\n{Fore.YELLOW}Checking environment setup...{Style.RESET_ALL}")

    issues = []
    warnings = []

    # Python version
    if sys.version_info < (3, 11):
        issues.append(f"Python 3.11+ required (current: {sys.version.split()[0]})")
    else:
        print(f"  Detected Python version: {sys.version.split()[0]}")

    # CUDA
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  CUDA is available. Device: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"    VRAM: {vram:.1f} GB")
        else:
            warnings.append("CUDA not available - will use CPU (much slower)")
    except ImportError:
        issues.append("PyTorch not installed")

    # Ollama API
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.get(f"{ollama_host}/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"  Connected to Ollama API at {ollama_host} (v{version})")

            # Check models
            models_resp = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if models_resp.status_code == 200:
                model_names = [m["name"] for m in models_resp.json().get("models", [])]
                required = ["qwen3.5:35b", "qwen3-vl:32b"]
                missing = [m for m in required if m not in model_names]
                if missing:
                    warnings.append(f"Missing Ollama models: {', '.join(missing)}")
                else:
                    print("  All required Ollama models are available.")
        else:
            issues.append(f"Ollama API not responding at {ollama_host}")
    except requests.RequestException:
        issues.append(f"Ollama API not accessible at {ollama_host}. Is Docker/Ollama running?")

    # DeepSeek-OCR venv
    deepseek_venv = Path("DeepSeek-OCR/.venv")
    if not deepseek_venv.exists():
        issues.append("DeepSeek-OCR venv not found - run ./setup_environments.sh")
    else:
        print(f"  Found DeepSeek-OCR environment at: {deepseek_venv}")

    # DeepSeek-OCR code
    config_path = Path("DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
    if not config_path.parent.exists():
        issues.append("DeepSeek-OCR code not found - run ./setup_environments.sh")
    else:
        print("  Found DeepSeek-OCR source code.")

    # Create required directories
    for d in ["outputs", "logs", "temp"]:
        Path(d).mkdir(exist_ok=True)

    # Summary
    print()
    if warnings:
        print(f"{Fore.YELLOW}Warnings during environment check:{Style.RESET_ALL}")
        for w in warnings:
            print(f"  • {w}")
        print()

    if issues:
        print(f"{Fore.RED}Critical issues found during environment check:{Style.RESET_ALL}")
        for i in issues:
            print(f"  • {i}")
        return False

    print(f"{Fore.GREEN}Environment check passed successfully!{Style.RESET_ALL}\n")
    return True


def process_pdf(pdf_path: str, config_path: str = None) -> dict:
    """Process PDF through the complete pipeline."""
    logging.info(f"Processing: {pdf_path}")

    from workflow.langgraph_workflow import run_workflow

    try:
        result = run_workflow(pdf_path, config_path)
        return result
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ExMat AI - Automated Battery Material Data Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf paper.pdf
  python main.py --pdf paper.pdf --verbose
  python main.py --pdf paper.pdf --skip-check
        """,
    )

    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-check", action="store_true", help="Skip environment checks")

    args = parser.parse_args()

    # Banner
    print_banner()

    # Verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Environment check
    if not args.skip_check:
        if not check_environment():
            print(f"\n{Fore.YELLOW}Run with --skip-check to bypass checks{Style.RESET_ALL}")
            sys.exit(1)

    # Validate PDF
    if not os.path.exists(args.pdf):
        print(f"\n{Fore.RED}Error: PDF file not found: {args.pdf}{Style.RESET_ALL}")
        sys.exit(1)

    # Process
    try:
        print(f"\n{'=' * 70}")
        print(f"{Fore.CYAN}Starting ExMat AI Extraction Pipeline{Style.RESET_ALL}")
        print(f"{'=' * 70}\n")
        print(f"Input PDF: {args.pdf}\n")

        result = process_pdf(args.pdf, args.config)

        print(f"\n{'=' * 70}")
        print(f"{Fore.GREEN}EXTRACTION COMPLETED SUCCESSFULLY!{Style.RESET_ALL}")
        print(f"{'=' * 70}\n")

        if result.get("output_file"):
            print(f"Output saved to: {Fore.CYAN}{result['output_file']}{Style.RESET_ALL}")

    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW} Process interrupted by user{Style.RESET_ALL}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"{Fore.RED}EXTRACTION FAILED!{Style.RESET_ALL}")
        print(f"{'=' * 70}\n")
        print(f"Error: {e}")
        logging.error("Fatal error occurred", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
