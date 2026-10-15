import pandas as pd
import requests
import time
from pathlib import Path
from collections import defaultdict

LABEL_FILE = Path("data/processed/genome_labels.csv")
OUTPUT_DIR = Path("data/raw/genomes")
MANIFEST_FILE = Path("data/processed/sequence_manifest.csv")

BASE_URL = "https://www.bv-brc.org/api/genome_sequence/"

BATCH_SIZE = 100
MAX_RETRIES = 3
TIMEOUT = 600

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LABEL_FILE)

# Normalize IDs
df["Genome ID"] = df["Genome ID"].astype(str).str.strip()

label_map = dict(
    zip(df["Genome ID"], df["phenotype"])
)

genome_ids = list(df["Genome ID"])

print("=" * 60)
print("BV-BRC BULK GENOME DOWNLOADER")
print("=" * 60)
print(f"Total genomes: {len(genome_ids):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Expected batches: {(len(genome_ids) + BATCH_SIZE - 1) // BATCH_SIZE}")

session = requests.Session()

manifest = {}

# ---------------------------------------------------------
# Detect already downloaded genomes
# ---------------------------------------------------------

for genome_id in genome_ids:

    output_file = OUTPUT_DIR / f"{genome_id}.fna"

    if output_file.exists() and output_file.stat().st_size > 0:

        manifest[genome_id] = {
            "genome_id": genome_id,
            "phenotype": label_map[genome_id],
            "status": "already_exists",
            "bytes": output_file.stat().st_size,
            "file": str(output_file)
        }


def save_manifest():

    rows = list(manifest.values())

    pd.DataFrame(rows).to_csv(
        MANIFEST_FILE,
        index=False
    )


# ---------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------

def parse_fasta_by_genome(fasta_text):

    sequences = defaultdict(list)

    current_genome = None
    current_sequence = []

    def flush():

        if current_genome is not None and current_sequence:

            sequences[current_genome].append(
                "".join(current_sequence)
            )

    for line in fasta_text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):

            flush()

            current_sequence = []

            # BV-BRC headers contain:
            #
            # [Escherichia coli ... | 562.123456]
            #
            if "[" in line and "|" in line and "]" in line:

                inside = line.rsplit("|", 1)[-1]

                if "]" in inside:
                    genome_id = inside.split("]", 1)[0].strip()
                else:
                    genome_id = None

                current_genome = genome_id

            else:
                current_genome = None

        else:

            current_sequence.append(line)

    flush()

    return sequences


# ---------------------------------------------------------
# Determine which genomes remain
# ---------------------------------------------------------

remaining = [
    genome_id
    for genome_id in genome_ids
    if genome_id not in manifest
]

print(f"\nAlready downloaded: {len(genome_ids) - len(remaining):,}")
print(f"Remaining: {len(remaining):,}")

# ---------------------------------------------------------
# Process batches
# ---------------------------------------------------------

for batch_start in range(0, len(remaining), BATCH_SIZE):

    batch = remaining[
        batch_start:batch_start + BATCH_SIZE
    ]

    batch_number = (
        batch_start // BATCH_SIZE
    ) + 1

    total_batches = (
        len(remaining) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("\n" + "=" * 60)
    print(
        f"BATCH {batch_number}/{total_batches} "
        f"({len(batch)} genomes)"
    )
    print("=" * 60)

    # BV-BRC RQL bulk query
    rql = (
        "in(genome_id,("
        + ",".join(batch)
        + "))"
        "&limit(100000)"
    )

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"Request attempt "
                f"{attempt}/{MAX_RETRIES}..."
            )

            response = session.post(
                BASE_URL,
                data=rql,
                headers={
                    "Content-Type":
                        "application/x-www-form-urlencoded",
                    "Accept":
                        "application/dna+fasta",
                },
                timeout=TIMEOUT,
            )

            print(
                f"HTTP {response.status_code} | "
                f"{len(response.content):,} bytes"
            )

            if response.status_code != 200:

                print(
                    response.text[:500]
                )

                if attempt < MAX_RETRIES:
                    time.sleep(5 * attempt)

                continue

            fasta_text = response.text

            if not fasta_text.strip():

                print("WARNING: Empty response")

                if attempt < MAX_RETRIES:
                    time.sleep(5 * attempt)

                continue

            # Parse FASTA
            sequences = parse_fasta_by_genome(
                fasta_text
            )

            print(
                f"Genome IDs found in response: "
                f"{len(sequences)}"
            )

            # -------------------------------------------------
            # Save one FASTA file per genome
            # -------------------------------------------------

            saved = 0

            for genome_id in batch:

                records = sequences.get(
                    genome_id,
                    []
                )

                if not records:

                    manifest[genome_id] = {
                        "genome_id": genome_id,
                        "phenotype":
                            label_map[genome_id],
                        "status": "no_sequence",
                        "bytes": 0,
                        "file": ""
                    }

                    continue

                output_file = (
                    OUTPUT_DIR /
                    f"{genome_id}.fna"
                )

                with open(
                    output_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    # Recreate FASTA records.
                    # Sequence boundaries don't matter for
                    # genome-level k-mer extraction because
                    # the downstream parser will treat these
                    # as separate contigs.

                    for idx, sequence in enumerate(
                        records,
                        start=1
                    ):

                        f.write(
                            f">{genome_id}.contig{idx}\n"
                        )

                        # Wrap sequence at 80 characters
                        for pos in range(
                            0,
                            len(sequence),
                            80
                        ):

                            f.write(
                                sequence[
                                    pos:pos + 80
                                ] + "\n"
                            )

                size = output_file.stat().st_size

                manifest[genome_id] = {
                    "genome_id": genome_id,
                    "phenotype":
                        label_map[genome_id],
                    "status": "downloaded",
                    "bytes": size,
                    "file": str(output_file)
                }

                saved += 1

            print(
                f"Saved genomes in batch: "
                f"{saved}/{len(batch)}"
            )

            save_manifest()

            success = True
            break

        except requests.RequestException as e:

            print(
                f"Request error: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

        except Exception as e:

            print(
                f"Unexpected error: {e}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    if not success:

        print(
            f"FAILED BATCH {batch_number}"
        )

        for genome_id in batch:

            if genome_id not in manifest:

                manifest[genome_id] = {
                    "genome_id": genome_id,
                    "phenotype":
                        label_map[genome_id],
                    "status": "batch_failed",
                    "bytes": 0,
                    "file": ""
                }

        save_manifest()

    # Small pause between batches
    time.sleep(1)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

save_manifest()

manifest_df = pd.DataFrame(
    list(manifest.values())
)

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(
    manifest_df["status"]
    .value_counts()
    .to_string()
)

downloaded = (
    manifest_df["status"]
    .isin(["downloaded", "already_exists"])
    .sum()
)

print(
    f"\nUsable genome FASTA files: "
    f"{downloaded:,}/{len(genome_ids):,}"
)

print(
    f"Manifest: {MANIFEST_FILE}"
)