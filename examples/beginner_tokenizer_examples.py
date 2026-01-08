#!/usr/bin/env python3
"""
Beginner-Friendly Tokenizer Examples

This script demonstrates how to use different tokenizers in a simple,
easy-to-understand way. Perfect for beginners!
"""

from llm_trainer.tokenizer import (
    create_tokenizer,
    get_available_tokenizers
)


def example_1_simple_tokenizer():
    """Example 1: The Simplest Tokenizer (Perfect for Beginners)"""
    print("=" * 60)
    print("Example 1: Simple Whitespace Tokenizer")
    print("=" * 60)
    print()
    
    # Create the simplest tokenizer
    tokenizer = create_tokenizer("simple")
    
    # Train it on some text
    texts = [
        "Hello world! This is a simple example.",
        "Tokenization is the process of breaking text into tokens.",
        "This tokenizer splits on whitespace - very easy to understand!"
    ]
    
    print("Training tokenizer on sample texts...")
    tokenizer.train(texts, verbose=True)
    print()
    
    # Test encoding
    test_text = "Hello world! This is a test."
    print(f"Original text: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Test decoding
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded}")
    print()


def example_2_character_tokenizer():
    """Example 2: Character-Level Tokenizer"""
    print("=" * 60)
    print("Example 2: Character-Level Tokenizer")
    print("=" * 60)
    print()
    
    # Create character tokenizer
    tokenizer = create_tokenizer("char")
    
    # Train on sample texts
    texts = [
        "Hello world!",
        "Character tokenization treats each character as a token.",
        "Very simple but creates long sequences!"
    ]
    
    print("Training character tokenizer...")
    tokenizer.train(texts, verbose=True)
    print()
    
    # Test
    test_text = "Hi!"
    print(f"Original text: {test_text}")
    
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()


def example_3_bpe_tokenizer():
    """Example 3: BPE Tokenizer (Most Common)"""
    print("=" * 60)
    print("Example 3: BPE Tokenizer (Recommended)")
    print("=" * 60)
    print()
    
    # Create BPE tokenizer
    tokenizer = create_tokenizer("bpe")
    
    # Train on sample texts
    texts = [
        "Byte Pair Encoding is a popular tokenization method.",
        "It learns to merge frequent character pairs.",
        "This creates a good balance between vocabulary size and sequence length."
    ] * 10  # Repeat to have enough data for BPE
    
    print("Training BPE tokenizer...")
    tokenizer.train(texts, vocab_size=1000, verbose=True)
    print()
    
    # Test
    test_text = "Byte Pair Encoding is efficient!"
    print(f"Original text: {test_text}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()


def example_4_using_factory():
    """Example 4: Using the Factory Function (Easiest Way)"""
    print("=" * 60)
    print("Example 4: Using Factory Function")
    print("=" * 60)
    print()
    
    # List available tokenizers
    available = get_available_tokenizers()
    print("Available tokenizers:")
    for tokenizer_type, description in list(available.items())[:3]:
        print(f"  - {tokenizer_type}: {description[:50]}...")
    print()
    
    # Create a tokenizer
    tokenizer = create_tokenizer("simple")
    
    # Train
    texts = ["This is easy!", "The factory makes it simple."]
    tokenizer.train(texts, verbose=True)
    
    # Use
    result = tokenizer.encode("This is easy!")
    print(f"Encoded: {result}")
    print()


def example_5_comparing_tokenizers():
    """Example 5: Comparing Different Tokenizers"""
    print("=" * 60)
    print("Example 5: Comparing Tokenizers")
    print("=" * 60)
    print()
    
    test_text = "Hello world! This is a comparison."
    
    tokenizers_to_test = ["simple", "char", "bpe"]
    
    for tokenizer_type in tokenizers_to_test:
        print(f"\n{tokenizer_type.upper()} Tokenizer:")
        print("-" * 40)
        
        tokenizer = create_tokenizer(tokenizer_type)
        
        # Train on same data
        texts = [
            "Hello world!",
            "This is a test.",
            "Comparing different tokenizers."
        ] * 5
        
        tokenizer.train(texts, vocab_size=500, verbose=False)
        
        # Test
        token_ids = tokenizer.encode(test_text, add_special_tokens=False)
        print(f"  Text: {test_text}")
        print(f"  Tokens: {len(token_ids)}")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Token IDs: {token_ids[:10]}...")  # Show first 10
    print()


def example_6_saving_and_loading():
    """Example 6: Saving and Loading Tokenizers"""
    print("=" * 60)
    print("Example 6: Saving and Loading Tokenizers")
    print("=" * 60)
    print()
    
    # Create and train tokenizer
    tokenizer = create_tokenizer("simple")
    texts = ["Save me!", "Load me later."]
    tokenizer.train(texts, verbose=False)
    
    # Save
    save_path = "./saved_tokenizer"
    print(f"Saving tokenizer to {save_path}...")
    tokenizer.save_pretrained(save_path)
    print("Saved!")
    print()
    
    # Load
    print(f"Loading tokenizer from {save_path}...")
    loaded_tokenizer = create_tokenizer("simple", pretrained_path=save_path)
    print("Loaded!")
    print()
    
    # Test that it works
    test_text = "Save me!"
    original_ids = tokenizer.encode(test_text)
    loaded_ids = loaded_tokenizer.encode(test_text)
    
    print(f"Original tokenizer: {original_ids}")
    print(f"Loaded tokenizer: {loaded_ids}")
    print(f"Match: {original_ids == loaded_ids}")
    print()


def example_7_available_tokenizers():
    """Example 7: See All Available Tokenizers"""
    print("=" * 60)
    print("Example 7: Available Tokenizers")
    print("=" * 60)
    print()
    
    available = get_available_tokenizers()
    
    print("Available tokenizer types:")
    print()
    for tokenizer_type, description in available.items():
        print(f"  • {tokenizer_type.upper()}")
        print(f"    {description}")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("BEGINNER-FRIENDLY TOKENIZER EXAMPLES")
    print("=" * 60)
    print("\nThese examples show you how to use different tokenizers")
    print("in a simple, easy-to-understand way.\n")
    
    try:
        example_1_simple_tokenizer()
        example_2_character_tokenizer()
        example_3_bpe_tokenizer()
        example_4_using_factory()
        example_5_comparing_tokenizers()
        example_6_saving_and_loading()
        example_7_available_tokenizers()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Try modifying the examples")
        print("  2. Train on your own text data")
        print("  3. Experiment with different tokenizer types")
        print("  4. Read the documentation for more advanced features")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you encounter errors, make sure you have installed")
        print("all required dependencies: pip install -e .")


if __name__ == "__main__":
    main()
