import pandas as pd
import requests
from pathlib import Path

LABEL_FILE = Path("data/processed/genome_labels.csv")

df = pd.read_csv(LABEL_FILE)

# Use 100 IDs that haven't necessarily been downloaded yet
genome_ids = [
    str(x).strip()
    for x in df["Genome ID"].head(100)
]

# RQL syntax supported by BV-BRC:
# in(genome_id,(ID1,ID2,...))
rql = "in(genome_id,(" + ",".join(genome_ids) + "))"

url = "https://www.bv-brc.org/api/genome_sequence/"

print(f"Requesting sequences for {len(genome_ids)} genomes...")
print(f"RQL length: {len(rql):,} characters")

try:
    response = requests.post(
        url,
        data=rql,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/dna+fasta",
        },
        timeout=300,
    )

    print("HTTP status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Response size:", len(response.content))

    if response.status_code != 200:
        print("\nERROR RESPONSE:")
        print(response.text[:2000])
        raise SystemExit(1)

    output = Path("data/raw/bulk_test.fna")
    output.write_bytes(response.content)

    print(f"\nSaved: {output}")
    print(f"Size: {output.stat().st_size:,} bytes")

    # Show the first few FASTA headers
    print("\nFirst FASTA headers:")

    count = 0

    for line in response.text.splitlines():
        if line.startswith(">"):
            print(line[:250])
            count += 1

            if count >= 10:
                break

    print(f"\nFASTA records shown: {count}")

except requests.RequestException as e:
    print("REQUEST ERROR:", e)