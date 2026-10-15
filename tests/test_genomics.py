"""Unit tests for genomic data processing, FASTA parsing, and k-mer extraction."""

import unittest
from pathlib import Path
import numpy as np

from amr_guard.data.fasta import parse_fasta, parse_fasta_str
from amr_guard.data.kmer import extract_kmers_from_contigs, get_kmer_vocabulary, extract_kmers_from_sequence
from amr_guard.data.loader import load_genome_dataset
from amr_guard.data.preprocess import check_data_leakage, prepare_features


class TestGenomicsPipeline(unittest.TestCase):

    def test_kmer_vocabulary_length_and_order(self):
        vocab = get_kmer_vocabulary(k=6)
        self.assertEqual(len(vocab), 4096)
        self.assertEqual(vocab[0], "AAAAAA")
        self.assertEqual(vocab[-1], "TTTTTT")
        self.assertEqual(len(set(vocab)), 4096)
        self.assertEqual(vocab, sorted(vocab))

    def test_fasta_parsing_multicontig(self):
        fasta_content = ">contig1\nACGTACGT\n>contig2\nGGCCGGCC\n"
        contigs, total_len, count = parse_fasta_str(fasta_content)
        self.assertEqual(count, 2)
        self.assertEqual(total_len, 16)
        self.assertEqual(contigs, ["ACGTACGT", "GGCCGGCC"])

    def test_multicontig_kmer_extraction_prevents_junction_kmers(self):
        # contig 1 ends with AAAAAA, contig 2 starts with TTTTTT
        contig1 = "AAAAAA"
        contig2 = "TTTTTT"

        # Separate contigs extraction
        freqs_separated = extract_kmers_from_contigs([contig1, contig2], k=6, normalize=False)
        vocab = get_kmer_vocabulary(k=6)
        idx_AAAAAA = vocab.index("AAAAAA")
        idx_TTTTTT = vocab.index("TTTTTT")

        self.assertEqual(freqs_separated[idx_AAAAAA], 1.0)
        self.assertEqual(freqs_separated[idx_TTTTTT], 1.0)
        # Sum of counts across all 4096 k-mers should be exactly 2
        self.assertEqual(freqs_separated.sum(), 2.0)

        # If concatenated into single string "AAAAAATTTTTT", artificial k-mers like AAAAAT would be created
        freqs_concatenated = extract_kmers_from_sequence("AAAAAATTTTTT", k=6, normalize=False)
        self.assertEqual(freqs_concatenated.sum(), 7.0)
        self.assertLess(freqs_separated.sum(), freqs_concatenated.sum())

    def test_invalid_base_handling(self):
        # Sequence with N base
        seq = "AAAAAANCCCCCC"
        freqs = extract_kmers_from_sequence(seq, k=6, normalize=False)
        vocab = get_kmer_vocabulary(k=6)
        idx_AAAAAA = vocab.index("AAAAAA")
        idx_CCCCCC = vocab.index("CCCCCC")

        self.assertEqual(freqs[idx_AAAAAA], 1.0)
        self.assertEqual(freqs[idx_CCCCCC], 1.0)
        # Any 6-mer overlapping 'N' must be ignored, so sum is exactly 2
        self.assertEqual(freqs.sum(), 2.0)

    def test_normalization(self):
        contigs = ["AAAAAA", "CCCCCC"]
        freqs = extract_kmers_from_contigs(contigs, k=6, normalize=True)
        self.assertAlmostEqual(freqs.sum(), 1.0)

    def test_data_leakage_detection(self):
        train_ids = {"g1", "g2", "g3"}
        val_ids = {"g4", "g5"}
        test_ids = {"g2", "g6"}

        with self.assertRaises(ValueError):
            check_data_leakage(train_ids, val_ids, test_ids)

    def test_prepare_features_dimensions(self):
        seq = "ACGTACGTACGTACGTACGTACGT"
        df = prepare_features(seq, k=6)
        self.assertEqual(df.shape, (1, 4096))
        self.assertAlmostEqual(df.iloc[0].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
