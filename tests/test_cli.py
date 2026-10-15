"""Unit tests for the CLI batch prediction command."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd
from click.testing import CliRunner

from amr_guard.cli.main import cli, OPENPYXL_AVAILABLE


class TestCliCommands(unittest.TestCase):

    def test_cli_predict_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "samples.csv"
            output_csv = Path(tmpdir) / "results.csv"

            df = pd.DataFrame([
                {"sequence": "ACGTACGTACGTACGTACGTACGT"},
                {"sequence": "GGCCGGCCGGCCGGCCGGCCGGCC"},
            ])
            df.to_csv(input_csv, index=False)

            runner = CliRunner()
            result = runner.invoke(cli, ["predict", "--input", str(input_csv), "--output", str(output_csv)])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(output_csv.exists())

            out_df = pd.read_csv(output_csv)
            self.assertIn("prediction", out_df.columns)
            self.assertIn("probability", out_df.columns)
            self.assertEqual(len(out_df), 2)


if __name__ == "__main__":
    unittest.main()
