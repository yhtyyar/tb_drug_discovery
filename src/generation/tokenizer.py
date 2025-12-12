"""SMILES tokenization for sequence-based molecular generation.

This module provides tokenization utilities for converting SMILES strings
to sequences of tokens suitable for neural network processing.

Supports:
- Character-level tokenization
- Atom-level tokenization with multi-character tokens (Cl, Br, etc.)
- Special tokens (PAD, START, END, UNK)
- Vocabulary creation from datasets

Example:
    >>> tokenizer = SmilesTokenizer()
    >>> tokens = tokenizer.tokenize("CCO")
    >>> indices = tokenizer.encode("CCO")
    >>> smiles = tokenizer.decode(indices)
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


# Regex pattern for SMILES tokenization (handles multi-character atoms)
SMILES_REGEX = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


# Common SMILES tokens
COMMON_TOKENS = [
    # Special tokens
    '<PAD>', '<START>', '<END>', '<UNK>',
    # Single atoms
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'I',
    # Multi-character atoms
    'Cl', 'Br',
    # Bonds
    '-', '=', '#', ':', '~',
    # Branches and rings
    '(', ')', '[', ']',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '%10', '%11', '%12',
    # Stereochemistry
    '/', '\\', '@', '@@',
    # Charges and others
    '+', '.', '*',
]


class SmilesTokenizer:
    """Tokenizer for SMILES strings.
    
    Converts SMILES strings to sequences of integer indices and vice versa.
    Supports both character-level and atom-level tokenization.
    
    Args:
        vocab: Optional vocabulary dictionary {token: index}.
        max_length: Maximum sequence length (for padding).
        
    Attributes:
        vocab: Token to index mapping.
        inv_vocab: Index to token mapping.
        vocab_size: Number of tokens in vocabulary.
        
    Example:
        >>> tokenizer = SmilesTokenizer()
        >>> tokenizer.fit(smiles_list)
        >>> encoded = tokenizer.encode("CCO", add_special=True)
        >>> decoded = tokenizer.decode(encoded)
    """
    
    # Special token indices
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 120,
    ):
        self.max_length = max_length
        self.regex = re.compile(SMILES_REGEX)
        
        if vocab is None:
            # Initialize with special tokens
            self.vocab = {
                '<PAD>': self.PAD_IDX,
                '<START>': self.START_IDX,
                '<END>': self.END_IDX,
                '<UNK>': self.UNK_IDX,
            }
        else:
            self.vocab = vocab
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into tokens.
        
        Args:
            smiles: SMILES string.
            
        Returns:
            List of tokens.
        """
        tokens = self.regex.findall(smiles)
        return tokens
    
    def fit(self, smiles_list: List[str], min_freq: int = 1) -> 'SmilesTokenizer':
        """Build vocabulary from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings.
            min_freq: Minimum token frequency to include.
            
        Returns:
            Self for chaining.
        """
        # Count token frequencies
        token_counts = {}
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add tokens meeting frequency threshold
        for token, count in sorted(token_counts.items()):
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        logger.info(f"Vocabulary built: {self.vocab_size} tokens")
        return self
    
    def encode(
        self,
        smiles: str,
        add_special: bool = True,
        pad: bool = True,
    ) -> List[int]:
        """Encode SMILES to sequence of indices.
        
        Args:
            smiles: SMILES string.
            add_special: Add START and END tokens.
            pad: Pad to max_length.
            
        Returns:
            List of token indices.
        """
        tokens = self.tokenize(smiles)
        
        # Convert to indices
        indices = [self.vocab.get(t, self.UNK_IDX) for t in tokens]
        
        # Add special tokens
        if add_special:
            indices = [self.START_IDX] + indices + [self.END_IDX]
        
        # Truncate if too long
        if len(indices) > self.max_length:
            indices = indices[:self.max_length - 1] + [self.END_IDX]
        
        # Pad
        if pad:
            indices = indices + [self.PAD_IDX] * (self.max_length - len(indices))
        
        return indices
    
    def decode(
        self,
        indices: List[int],
        remove_special: bool = True,
    ) -> str:
        """Decode sequence of indices to SMILES.
        
        Args:
            indices: List of token indices.
            remove_special: Remove special tokens from output.
            
        Returns:
            SMILES string.
        """
        tokens = []
        for idx in indices:
            if idx == self.PAD_IDX:
                continue
            if remove_special and idx in (self.START_IDX, self.END_IDX):
                continue
            token = self.inv_vocab.get(idx, '')
            if token:
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(
        self,
        smiles_list: List[str],
        add_special: bool = True,
    ) -> np.ndarray:
        """Encode batch of SMILES.
        
        Args:
            smiles_list: List of SMILES strings.
            add_special: Add special tokens.
            
        Returns:
            2D numpy array of shape (batch_size, max_length).
        """
        encoded = [self.encode(s, add_special=add_special) for s in smiles_list]
        return np.array(encoded, dtype=np.int64)
    
    def batch_decode(
        self,
        indices_batch: np.ndarray,
        remove_special: bool = True,
    ) -> List[str]:
        """Decode batch of sequences.
        
        Args:
            indices_batch: 2D array of indices.
            remove_special: Remove special tokens.
            
        Returns:
            List of SMILES strings.
        """
        return [self.decode(indices, remove_special) for indices in indices_batch]
    
    def save(self, path: str) -> None:
        """Save tokenizer vocabulary."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'max_length': self.max_length,
            }, f, indent=2)
        logger.info(f"Tokenizer saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SmilesTokenizer':
        """Load tokenizer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab=data['vocab'], max_length=data['max_length'])
        logger.info(f"Tokenizer loaded: {path}")
        return tokenizer


def create_vocabulary(
    smiles_list: List[str],
    min_freq: int = 1,
    max_length: int = 120,
) -> SmilesTokenizer:
    """Create tokenizer with vocabulary from SMILES list.
    
    Args:
        smiles_list: List of SMILES strings.
        min_freq: Minimum token frequency.
        max_length: Maximum sequence length.
        
    Returns:
        Fitted SmilesTokenizer.
    """
    tokenizer = SmilesTokenizer(max_length=max_length)
    tokenizer.fit(smiles_list, min_freq=min_freq)
    return tokenizer


def get_smiles_statistics(smiles_list: List[str]) -> Dict:
    """Get statistics about SMILES dataset.
    
    Args:
        smiles_list: List of SMILES strings.
        
    Returns:
        Dictionary of statistics.
    """
    tokenizer = SmilesTokenizer()
    
    lengths = []
    all_tokens = set()
    
    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        lengths.append(len(tokens))
        all_tokens.update(tokens)
    
    return {
        'num_molecules': len(smiles_list),
        'unique_tokens': len(all_tokens),
        'avg_length': np.mean(lengths),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'std_length': np.std(lengths),
        'p95_length': np.percentile(lengths, 95),
    }
