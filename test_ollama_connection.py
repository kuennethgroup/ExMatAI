"""
Test Ollama connection to Docker container
"""

import ollama
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

def test_ollama_connection():
    """Test connection to Ollama Docker container"""
    
    print("🧪 Testing Ollama Connection")
    print("="*80)
    print(f"Host: {OLLAMA_HOST}")
    print("")
    
    try:
        # Create client
        client = ollama.Client(host=OLLAMA_HOST)
        
        # List models
        print("📦 Available models:")
        models = client.list()
        
        for model in models.get('models', []):
            print(f"  ✓ {model['name']}")
        
        print("")
        
        # Test simple generation
        print("🧪 Testing model generation...")
        response = client.generate(
            model='qwen3-vl:8b',
            prompt='Say "Hello from Docker Ollama!"'
        )
        
        print(f"Response: {response['response']}")
        print("")
        print("✅ Ollama connection test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Ollama connection test FAILED!")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama_connection()
