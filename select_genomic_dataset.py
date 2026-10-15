import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/BVBRC_genome_amr.csv")
OUTPUT = Path("data/processed/genome_labels.csv")

RANDOM_STATE = 42
N_PER_CLASS = 2500

print("Loading AMR dataset...")
df = pd.read_csv(INPUT, low_memory=False)

# ---------------------------------------------------------
# 1. Restrict to E. coli + ciprofloxacin + binary phenotype
# ---------------------------------------------------------

df = df[
    (df["Taxon ID"] == 562)
    & (df["Antibiotic"].astype(str).str.lower() == "ciprofloxacin")
    & (df["Resistant Phenotype"].isin(["Resistant", "Susceptible"]))
].copy()

print(f"Rows after E. coli + ciprofloxacin + binary phenotype filter: {len(df):,}")

# ---------------------------------------------------------
# 2. Keep only the columns we need
# ---------------------------------------------------------

df = df[["Genome ID", "Resistant Phenotype"]].copy()

df["Genome ID"] = df["Genome ID"].astype(str).str.strip()

# Remove missing IDs
df = df[df["Genome ID"].notna() & (df["Genome ID"] != "")]

# ---------------------------------------------------------
# 3. Detect genomes with conflicting labels
# ---------------------------------------------------------

label_counts = (
    df.groupby("Genome ID")["Resistant Phenotype"]
    .nunique()
)

conflicting_ids = label_counts[label_counts > 1].index

print(f"Conflicting genome IDs: {len(conflicting_ids):,}")

# Remove conflicting genomes completely.
# This is safer than arbitrarily assigning Resistant/Susceptible.
df = df[~df["Genome ID"].isin(conflicting_ids)].copy()

# ---------------------------------------------------------
# 4. Deduplicate
# ---------------------------------------------------------

df = df.drop_duplicates(subset=["Genome ID"])

print("\nAvailable genomes after conflict removal:")

print(
    df["Resistant Phenotype"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 5. Check that we have enough genomes
# ---------------------------------------------------------

counts = df["Resistant Phenotype"].value_counts()

if counts.get("Resistant", 0) < N_PER_CLASS:
    raise RuntimeError("Not enough Resistant genomes.")

if counts.get("Susceptible", 0) < N_PER_CLASS:
    raise RuntimeError("Not enough Susceptible genomes.")

# ---------------------------------------------------------
# 6. Balanced sampling
# ---------------------------------------------------------

resistant = (
    df[df["Resistant Phenotype"] == "Resistant"]
    .sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
)

susceptible = (
    df[df["Resistant Phenotype"] == "Susceptible"]
    .sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
)

selected = pd.concat(
    [resistant, susceptible],
    ignore_index=True
)

# Shuffle final dataset
selected = selected.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ---------------------------------------------------------
# 7. Rename phenotype column
# ---------------------------------------------------------

selected = selected.rename(
    columns={"Resistant Phenotype": "phenotype"}
)

# ---------------------------------------------------------
# 8. Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

selected.to_csv(OUTPUT, index=False)

print("\n========================================")
print("GENOMIC DATASET CREATED")
print("========================================")
print(f"Total genomes: {len(selected):,}")
print(
    selected["phenotype"]
    .value_counts()
    .to_string()
)
print(f"\nSaved to: {OUTPUT}")
print("\nFirst 10 rows:")
print(selected.head(10).to_string(index=False))