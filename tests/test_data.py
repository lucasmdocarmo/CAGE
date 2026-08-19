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


# The old skipif(True) test_squad_v2_loader placeholder was DELETED (task #141,
# K-led7): it could never run (unconditional skip counted as "environmental" in
# the suite headline) and the real squad_v2 loader behavior is pinned offline in
# tests/test_dataset_loaders.py against schema-faithful synthetic rows.
