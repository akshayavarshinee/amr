"""Genome-level k-mer feature extraction module.

Configuration: k = 6 -> 4^6 = 4096 possible DNA 6-mers.
- Uses all 4096 possible 6-mers (no reverse complement canonicalization).
- Deterministic fixed vocabulary ordering (alphabetical: AAAAAA ... TTTTTT).
- Multi-contig handling: Extracts k-mers within each contig independently (no artificial boundary k-mers),
  accumulates raw counts across all contigs for a genome, and then normalizes by total valid k-mers.
- Invalid bases: K-mers containing characters outside A, C, G, T are ignored.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd

VALID_BASES = set("ACGT")


def get_kmer_vocabulary(k: int = 6) -> list[str]:
    """Generates deterministic, alphabetically sorted list of all 4^k DNA k-mers."""
    return ["".join(p) for p in itertools.product(["A", "C", "G", "T"], repeat=k)]


def extract_kmers_from_contigs(
    contigs: Sequence[str],
    k: int = 6,
    vocabulary: list[str] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Extracts k-mer frequencies for a genome given a list of its contigs.

    Iterates through each contig independently to prevent artificial k-mers across contig boundaries.
    Ignores any k-mer containing characters outside A, C, G, T.
    Normalizes aggregate counts by total valid k-mers observed across all contigs.

    Returns:
        1D numpy array of length 4^k (4096 for k=6) ordered according to vocabulary.
    """
    vocab = vocabulary or get_kmer_vocabulary(k)
    vocab_to_idx = {kmer: idx for idx, kmer in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float64)

    total_valid_kmers = 0

    # Base to int encoding for fast lookup without string dicts
    BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

    for contig in contigs:
        c_len = len(contig)
        if c_len < k:
            continue
        
        # Check if we can use direct integer indexing (k=6)
        if k == 6 and vocabulary is None:
            for i in range(c_len - 5):
                sub = contig[i : i + 6]
                b0 = BASE_MAP.get(sub[0])
                if b0 is None: continue
                b1 = BASE_MAP.get(sub[1])
                if b1 is None: continue
                b2 = BASE_MAP.get(sub[2])
                if b2 is None: continue
                b3 = BASE_MAP.get(sub[3])
                if b3 is None: continue
                b4 = BASE_MAP.get(sub[4])
                if b4 is None: continue
                b5 = BASE_MAP.get(sub[5])
                if b5 is None: continue
                
                idx = (b0 << 10) | (b1 << 8) | (b2 << 6) | (b3 << 4) | (b4 << 2) | b5
                counts[idx] += 1.0
                total_valid_kmers += 1
        else:
            for i in range(c_len - k + 1):
                kmer = contig[i : i + k]
                if all(base in VALID_BASES for base in kmer):
                    idx = vocab_to_idx.get(kmer)
                    if idx is not None:
                        counts[idx] += 1.0
                        total_valid_kmers += 1

    if normalize and total_valid_kmers > 0:
        counts /= total_valid_kmers

    return counts


def extract_kmers_from_sequence(
    sequence: str,
    k: int = 6,
    vocabulary: list[str] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Convenience wrapper for a single sequence / single contig."""
    return extract_kmers_from_contigs([sequence], k=k, vocabulary=vocabulary, normalize=normalize)


def compute_kmer_matrix(
    genome_contigs: Sequence[Sequence[str]],
    k: int = 6,
    vocabulary: list[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Computes a (n_genomes, 4^k) k-mer frequency DataFrame for a list of genome contig lists."""
    vocab = vocabulary or get_kmer_vocabulary(k)
    matrix = np.zeros((len(genome_contigs), len(vocab)), dtype=np.float32)

    for idx, contigs in enumerate(genome_contigs):
        matrix[idx] = extract_kmers_from_contigs(contigs, k=k, vocabulary=vocab, normalize=normalize)

    return pd.DataFrame(matrix, columns=vocab)
