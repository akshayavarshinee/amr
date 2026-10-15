"""Phase 5: CLI batch processing.

Usage:
    amr-guard predict --input samples.csv --output results.csv
    amr-guard predict --input samples.csv --output results.xlsx
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from amr_guard.models.predict import load_model, predict_batch
from amr_guard.utils.config import load_params

try:
    import openpyxl  # noqa: F401

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".fasta", ".fa", ".fna"):
        return pd.DataFrame([{"fasta_path": str(path)}])
    if path.is_dir():
        fastas = [str(f) for ext in ("*.fasta", "*.fa", "*.fna") for f in path.glob(ext)]
        return pd.DataFrame([{"fasta_path": f} for f in fastas])
    if suffix in (".xlsx", ".xls"):
        if not OPENPYXL_AVAILABLE:
            raise click.ClickException("openpyxl not installed — cannot read Excel input.")
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_output(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in (".xlsx", ".xls"):
        if not OPENPYXL_AVAILABLE:
            raise click.ClickException("openpyxl not installed — cannot write Excel output.")
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


@click.group()
def cli() -> None:
    """AMR-GenomeGuard CLI."""


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="Path to FASTA file, CSV file, or directory of FASTAs")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path), help="Path to save output CSV or Excel file")
@click.option("--model-path", "model_path", default=None, type=click.Path(exists=True, path_type=Path), help="Custom model artifact path")
def predict(input_path: Path, output_path: Path, model_path: Path | None) -> None:
    """Predict AMR phenotype for a single FASTA file, a folder of FASTAs, or a CSV/Excel file."""
    config = load_params()["cli"]
    model = load_model(model_path)
    features_df = _read_input(input_path)

    if features_df.empty:
        raise click.ClickException(f"No valid sequence data found at {input_path}")

    chunk_size = config["chunk_size"]
    results = []
    input_records = features_df.to_dict(orient="records")
    for start in range(0, len(input_records), chunk_size):
        chunk = input_records[start : start + chunk_size]
        results.append(predict_batch(model, chunk))
        click.echo(f"Processed {min(start + chunk_size, len(input_records))}/{len(input_records)} samples")

    output_df = pd.concat(results, ignore_index=True)
    combined = pd.concat([features_df.reset_index(drop=True), output_df], axis=1)
    _write_output(combined, output_path)
    click.echo(f"Successfully wrote {len(combined)} predictions to {output_path}")


if __name__ == "__main__":
    cli()
