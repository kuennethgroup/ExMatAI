"""
Test DeepSeek-OCR wrapper
"""

from utils.deepseek_ocr_wrapper import DeepSeekOCRWrapper
import os

def test_deepseek_ocr():
    """Test DeepSeek-OCR processing"""
    
    # Test PDF
    pdf_path = "sample_data_paper.pdf"
    
    if not os.path.exists(pdf_path):
        print("❌ Test PDF not found")
        return
    
    print("🧪 Testing DeepSeek-OCR Wrapper")
    print("="*80)
    
    # Initialize wrapper
    wrapper = DeepSeekOCRWrapper()
    
    # Process PDF
    result = wrapper.process_pdf(pdf_path)
    
    # Check results
    print("\n📊 Results:")
    print(f"  ├─ Text length: {len(result['extracted_text'])} chars")
    print(f"  ├─ Pages: {len(result['text_by_page'])}")
    print(f"  ├─ Markdown file: {result['markdown_file']}")
    print(f"  └─ Images dir: {result['images_dir']}")
    
    # Print sample text
    print("\n📝 Sample extracted text:")
    print(result['extracted_text'][:500])
    print("...")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_deepseek_ocr()
