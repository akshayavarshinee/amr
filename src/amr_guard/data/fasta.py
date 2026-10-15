"""Robust FASTA file parser for genomic sequences.

Requirements:
- Ignores FASTA header lines starting with '>'
- Keeps contigs as separate strings (does NOT concatenate contigs into one string to avoid boundary k-mers)
- Converts all bases to uppercase
- Reports total sequence length and contig count per genome
- Does not modify raw FASTA files
"""

from __future__ import annotations

from pathlib import Path


def parse_fasta(fasta_path: str | Path) -> tuple[list[str], int, int]:
    """Reads a FASTA file and returns (contigs_list, total_sequence_length, contig_count).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or contains no sequence data.
    """
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    contigs = []
    current_contig_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith(">"):
                if current_contig_lines:
                    contigs.append("".join(current_contig_lines))
                    current_contig_lines = []
            else:
                current_contig_lines.append(line_str.upper())

        if current_contig_lines:
            contigs.append("".join(current_contig_lines))

    if not contigs:
        raise ValueError(f"FASTA file {path} contains no valid sequence data.")

    contig_count = len(contigs)
    total_sequence_length = sum(len(c) for c in contigs)

    return contigs, total_sequence_length, contig_count


def parse_fasta_str(fasta_content: str) -> tuple[list[str], int, int]:
    """Parses a FASTA format string and returns (contigs_list, total_sequence_length, contig_count)."""
    contigs = []
    current_contig_lines = []

    for line in fasta_content.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith(">"):
            if current_contig_lines:
                contigs.append("".join(current_contig_lines))
                current_contig_lines = []
        else:
            current_contig_lines.append(line_str.upper())

    if current_contig_lines:
        contigs.append("".join(current_contig_lines))

    if not contigs:
        clean_seq = "".join(c.upper() for c in fasta_content.splitlines() if not c.startswith(">")).strip()
        if clean_seq:
            return [clean_seq], len(clean_seq), 1
        raise ValueError("FASTA content contains no valid sequence data.")

    total_sequence_length = sum(len(c) for c in contigs)
    return contigs, total_sequence_length, len(contigs)
