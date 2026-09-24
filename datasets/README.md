# Datasets

No data is included yet. Raw `.mcpl` and `.mcpl.gz` files are ignored by Git.

For each future dataset:

1. Copy `configs/datasets/dataset.template.yaml`.
2. Replace every `TODO` with confirmed information.
3. Put the local MCPL file under `datasets/reference/` or set another path.
4. Record its checksum, access conditions, units, filters, and scoring plane.

Planned categories are reactor, spallation, CANS/HBS, diffraction, SANS, INS,
and TOF. Do not guess missing metadata and do not commit restricted data.
