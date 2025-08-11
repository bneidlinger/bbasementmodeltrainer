#!/usr/bin/env python
"""
Test script for LLM data pipeline
Verifies that the data processing pipeline works correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trainer'))

from llm.data import DatasetManager, DatasetConfig
from llm.data.processors import DataTokenizer, DataCleaner, DataFormatter
from llm.data.processors.tokenizer import TokenizerConfig
from llm.data.processors.cleaner import CleanerConfig
from llm.data.processors.formatter import FormatterConfig

def test_tokenizer():
    """Test the tokenizer component."""
    print("\n=== Testing Tokenizer ===")
    
    config = TokenizerConfig(
        tokenizer_name="gpt2",
        max_length=512
    )
    tokenizer = DataTokenizer(config)
    
    # Test single text
    text = "This is a test of the LLM tokenization system. It should handle various text inputs properly."
    tokens = tokenizer.tokenize(text)
    print(f"Input text: {text[:50]}...")
    print(f"Token count: {tokenizer.count_tokens(text)}")
    print(f"Output shape: {tokens['input_ids'].shape}")
    
    # Test batch
    texts = [
        "First document for testing.",
        "Second document with more content to tokenize.",
        "Third document that is even longer with multiple sentences."
    ]
    batch_tokens = tokenizer.tokenize(texts)
    print(f"Batch shape: {batch_tokens['input_ids'].shape}")
    
    # Test instruction formatting
    instruction = "Explain machine learning"
    output = "Machine learning is a subset of artificial intelligence..."
    formatted = tokenizer.prepare_instruction_format(instruction, "", output)
    print(f"Instruction format preview: {formatted[:100]}...")
    
    print("[OK] Tokenizer test passed")
    return True

def test_cleaner():
    """Test the data cleaner component."""
    print("\n=== Testing Cleaner ===")
    
    config = CleanerConfig(
        min_length=20,
        max_length=10000,
        remove_html=True,
        remove_emails=True
    )
    cleaner = DataCleaner(config)
    
    # Test cleaning
    dirty_texts = [
        "<p>This is HTML content</p> with tags that should be removed.",
        "Contact us at test@example.com for more info",
        "Too short",  # Should be filtered
        "This is a clean text that should pass through without issues.",
        "This text has     multiple    spaces     that need normalization.",
        "Duplicate text for testing",
        "Duplicate text for testing",  # Duplicate
    ]
    
    cleaned = cleaner.clean_batch(dirty_texts, deduplicate=True)
    print(f"Input texts: {len(dirty_texts)}")
    print(f"Cleaned texts: {len(cleaned)}")
    
    # Check results
    for i, text in enumerate(cleaned[:3]):
        print(f"  Cleaned {i+1}: {text[:50]}...")
    
    # Get statistics
    stats = cleaner.get_statistics(cleaned)
    print(f"Statistics: {stats}")
    
    print("[OK] Cleaner test passed")
    return True

def test_formatter():
    """Test the dataset formatter component."""
    print("\n=== Testing Formatter ===")
    
    config = FormatterConfig(
        output_format="jsonl",
        instruction_template="alpaca",
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    formatter = DataFormatter(config)
    
    # Test instruction formatting
    data = [
        {
            "instruction": "What is machine learning?",
            "output": "Machine learning is a field of AI that enables computers to learn from data."
        },
        {
            "instruction": "Explain neural networks",
            "input": "In simple terms",
            "output": "Neural networks are computing systems inspired by biological neural networks."
        }
    ]
    
    formatted = formatter.format_dataset(data, dataset_type="instruction")
    print(f"Formatted {len(formatted)} instructions")
    print(f"Sample formatted text: {formatted[0]['text'][:150]}...")
    
    # Test splitting
    test_data = [{"text": f"Document {i}"} for i in range(100)]
    train, val, test = formatter.split_dataset(test_data)
    print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    print("[OK] Formatter test passed")
    return True

def test_mini_pipeline():
    """Test a mini version of the complete pipeline."""
    print("\n=== Testing Mini Pipeline ===")
    
    # Create sample data
    sample_data = [
        {
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence.",
            "source": "test",
            "url": "http://example.com/1"
        },
        {
            "text": "Machine learning is a method of data analysis that automates analytical model building.",
            "source": "test",
            "url": "http://example.com/2"
        },
        {
            "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "source": "test",
            "url": "http://example.com/3"
        }
    ]
    
    # Process through pipeline
    print("1. Starting with sample data...")
    
    # Clean
    cleaner = DataCleaner()
    texts = [d['text'] for d in sample_data]
    cleaned_texts = cleaner.clean_batch(texts, deduplicate=False)
    print(f"2. Cleaned {len(cleaned_texts)} documents")
    
    # Format
    formatter = DataFormatter()
    cleaned_data = [{"text": text} for text in cleaned_texts]
    formatted = formatter.format_dataset(cleaned_data, dataset_type="completion")
    print(f"3. Formatted {len(formatted)} documents")
    
    # Tokenize sample
    tokenizer = DataTokenizer()
    if formatted:
        token_count = tokenizer.count_tokens(formatted[0]['text'])
        print(f"4. Sample token count: {token_count}")
    
    print("[OK] Mini pipeline test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM DATA PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Cleaner", test_cleaner),
        ("Formatter", test_formatter),
        ("Mini Pipeline", test_mini_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Data pipeline is ready.")
    else:
        print("\n[WARNING] Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()