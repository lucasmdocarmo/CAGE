"""
Tests for data loading module.
"""

import pytest
from src.data.loader import CAGExample, get_loader


def test_cag_example_format_prompt():
    """Test CAGExample prompt formatting."""
    example = CAGExample(
        id="test1",
        question="What is the capital of France?",
        context=["France is a country in Europe.", "Paris is the capital."],
        answer="Paris",
        metadata={},
    )
    
    # Test with context
    prompt_with_context = example.format_prompt(include_context=True)
    assert "What is the capital of France?" in prompt_with_context
    assert "Context 1:" in prompt_with_context
    assert "France is a country" in prompt_with_context
    
    # Test without context
    prompt_without_context = example.format_prompt(include_context=False)
    assert "What is the capital of France?" in prompt_without_context
    assert "Context" not in prompt_without_context


def test_get_loader_invalid_dataset():
    """Test get_loader with invalid dataset name."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_loader("invalid_dataset")


def test_get_loader_valid_datasets():
    """Test get_loader returns correct loader types."""
    valid_datasets = ["hotpotqa", "qasper", "squad_v2", "trivia_qa"]
    
    for dataset_name in valid_datasets:
        loader = get_loader(dataset_name, split="validation")
        assert loader is not None
        assert hasattr(loader, "load")
        assert hasattr(loader, "sample")


# Note: Full dataset loading tests require network access and are slow
# These would be better as integration tests
@pytest.mark.slow
@pytest.mark.skipif(True, reason="Requires network access and dataset download")
def test_squad_v2_loader():
    """Test loading SQuAD v2 dataset (integration test)."""
    loader = get_loader("squad_v2", split="validation")
    examples = loader.load(max_examples=5)
    
    assert len(examples) <= 5
    assert all(isinstance(ex, CAGExample) for ex in examples)
    assert all(ex.question for ex in examples)
    assert all(ex.context for ex in examples)
