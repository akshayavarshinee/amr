"""Unit tests for the genomic preprocessing pipeline."""

import unittest
from pathlib import Path
import pandas as pd

from amr_guard.data.preprocess import check_data_leakage, prepare_features
from amr_guard.data.kmer import get_kmer_vocabulary


class TestPreprocessing(unittest.TestCase):

    def test_prepare_features_str_input(self):
        fasta_str = ">seq1\nACGTACGTACGTACGTACGTACGT"
        prepared = prepare_features(fasta_str, k=6)
        vocab = get_kmer_vocabulary(k=6)
        self.assertEqual(list(prepared.columns), vocab)
        self.assertEqual(len(prepared), 1)

    def test_prepare_features_dict_input(self):
        input_dict = {"sequence": "ACGTACGTACGTACGTACGTACGT"}
        prepared = prepare_features(input_dict, k=6)
        vocab = get_kmer_vocabulary(k=6)
        self.assertEqual(list(prepared.columns), vocab)
        self.assertEqual(len(prepared), 1)

    def test_data_leakage_assertion_passes(self):
        train_ids = {"g1", "g2", "g3"}
        val_ids = {"g4", "g5"}
        test_ids = {"g6", "g7"}
        # Should not raise exception
        check_data_leakage(train_ids, val_ids, test_ids)

    def test_data_leakage_assertion_fails(self):
        train_ids = {"g1", "g2", "g3"}
        val_ids = {"g3", "g4"}  # 'g3' overlap
        test_ids = {"g5"}
        with self.assertRaises(ValueError):
            check_data_leakage(train_ids, val_ids, test_ids)


if __name__ == "__main__":
    unittest.main()
