#!/usr/bin/env python3
"""
ExMat AI - Main Execution Script
Automated Battery Material Data Extraction from Scientific Papers
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/exmat_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}╔════════════════════════════════════════════════════════════════╗
║                          ExMat AI                              ║
║        Automated Battery Material Data Extraction              ║
║                                                                ║
║  Powered by: DeepSeek-OCR • Qwen3 • MolDetV2 • ChemVLM       ║
╚════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def check_environment():
    """Check if environment is properly setup"""
    print(f"{Fore.YELLOW}🔍 Checking environment...{Style.RESET_ALL}")
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required (current: {sys.version.split()[0]})")
    else:
        print(f"  ✓ Python version: {sys.version.split()[0]}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            warnings.append("CUDA not available - will use CPU (much slower)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # Check Ollama API (Docker or host)
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    try:
        response = requests.get(f"{ollama_host}/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"  ✓ Ollama API: {ollama_host}")
            print(f"    Version: {version_info.get('version', 'unknown')}")
            
            # Check if required models are available
            models_response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_names = [m['name'] for m in models_data.get('models', [])]
                
                required_models = ['qwen3-vl:8b']
                missing_models = [m for m in required_models if m not in model_names]
                
                if missing_models:
                    warnings.append(f"Missing Ollama models: {', '.join(missing_models)}")
                    print(f"  ⚠️  Missing models: {', '.join(missing_models)}")
                else:
                    print(f"  ✓ Required Ollama models available")
        else:
            issues.append(f"Ollama API not responding correctly at {ollama_host}")
    except requests.exceptions.RequestException as e:
        issues.append(f"Ollama API not accessible at {ollama_host}. Is Docker container running?")
        print(f"  ✗ Ollama check failed: {e}")
    
    # Check DeepSeek-OCR setup
    deepseek_venv = Path("DeepSeek-OCR/.venv")
    if not deepseek_venv.exists():
        issues.append("DeepSeek-OCR environment not setup - run ./setup_environments.sh")
    else:
        print(f"  ✓ DeepSeek-OCR environment: {deepseek_venv}")
    
    # Check DeepSeek-OCR code
    config_path = Path("DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
    if not config_path.parent.exists():
        issues.append("DeepSeek-OCR code not found - run ./setup_environments.sh")
    else:
        print(f"  ✓ DeepSeek-OCR code found")
    
    # Check required directories
    for dir_name in ["outputs", "logs", "temp"]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Print summary
    print("")
    
    if warnings:
        print(f"{Fore.YELLOW}⚠️  Warnings:{Style.RESET_ALL}")
        for warning in warnings:
            print(f"  • {warning}")
        print("")
    
    if issues:
        print(f"{Fore.RED}⚠️  Issues found:{Style.RESET_ALL}")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    print(f"{Fore.GREEN}✅ Environment check passed!{Style.RESET_ALL}\n")
    return True


    

def process_pdf(pdf_path: str, config_path: str = "config.yaml"):
    """Process a single PDF file"""
    from workflow.langgraph_workflow import run_workflow
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    logger.info(f"Processing: {pdf_path}")
    
    try:
        result = run_workflow(pdf_path, config_path)
        return result
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
        return None

def batch_process(directory: str, config_path: str = "config.yaml"):
    """Process all PDFs in a directory"""
    pdf_files = list(Path(directory).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(pdf_files)}: {pdf_path.name}")
        print(f"{'='*80}")
        
        result = process_pdf(str(pdf_path), config_path)
        results.append({
            'file': pdf_path.name,
            'success': result is not None
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Batch Processing Summary")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['success'])
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ExMat AI - Automated Battery Material Data Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF
  python main.py --pdf paper.pdf
  
  # Process all PDFs in directory
  python main.py --batch papers_folder/
  
  # Use custom config
  python main.py --pdf paper.pdf --config custom_config.yaml
  
  # Verbose output
  python main.py --pdf paper.pdf --verbose
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        help='Path to PDF file to process'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Directory containing PDF files for batch processing'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory (default: outputs/)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip environment check'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment
    if not args.skip_check:
        if not check_environment():
            print(f"\n{Fore.RED}Please fix environment issues before proceeding.{Style.RESET_ALL}")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process files
    if args.pdf:
        result = process_pdf(args.pdf, args.config)
        if result:
            print(f"\n{Fore.GREEN}✅ Processing completed successfully!{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"\n{Fore.RED}❌ Processing failed!{Style.RESET_ALL}")
            sys.exit(1)
    
    elif args.batch:
        batch_process(args.batch, args.config)
    
    else:
        parser.print_help()
        print(f"\n{Fore.YELLOW}Please specify --pdf or --batch{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️  Interrupted by user{Style.RESET_ALL}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Fatal error: {e}{Style.RESET_ALL}")
        logger.exception("Fatal error occurred")
        sys.exit(1)
