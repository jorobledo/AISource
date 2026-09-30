# Datasets

No data is stored in Git. Raw `.mcpl` and `.mcpl.gz` files are ignored by Git
and hosted on [ERDA](https://erda.dk) instead.

## Downloading the data from ERDA

The files are published through a public, read-only share link that needs no
ERDA account:

<https://sid.erda.dk/sharelink/dd7pdnWvOQ>

| File | Size | Contents |
|---|---|---|
| `d1b.mcpl.gz` | 1.8 GB | McStas 3.7.7 output of the `ILL_H22_D1B` instrument |

The share link itself opens a file listing in the browser. For the command
line, each file is served directly at
`https://sid.erda.dk/share_redirect/dd7pdnWvOQ/<filename>`.

Run the commands below from the repository root. They place the file in
`datasets/reference/`, which is ignored by Git.

```bash
mkdir -p datasets/reference
curl -fL -C - -o datasets/reference/d1b.mcpl.gz \
  https://sid.erda.dk/share_redirect/dd7pdnWvOQ/d1b.mcpl.gz
```

`-C -` resumes an interrupted download when the command is run again. The
equivalent with `wget` is:

```bash
wget -c -P datasets/reference \
  https://sid.erda.dk/share_redirect/dd7pdnWvOQ/d1b.mcpl.gz
```

Check the download against the expected SHA-256 checksum:

```bash
echo "69c6f51dc78e82e2a5d01d054a221ad1129b96c84c0eb4d153c9d2381c68f252  datasets/reference/d1b.mcpl.gz" \
  | shasum -a 256 -c -
```

On Linux `sha256sum -c -` can be used in place of `shasum -a 256 -c -`.

Keep the file compressed, since MCPL reads `.mcpl.gz` directly. With the
[MCPL](https://mctools.github.io/mcpl/) tools installed, the header and the
first particles can be inspected with:

```bash
mcpltool datasets/reference/d1b.mcpl.gz
```

Dataset configs refer to the data directory through `AISOURCE_DATA`:

```bash
export AISOURCE_DATA="$PWD/datasets/reference"
```

## Adding a dataset

For each future dataset:

1. Copy `configs/datasets/dataset.template.yaml`.
2. Replace every `TODO` with confirmed information.
3. Put the local MCPL file under `datasets/reference/` or set another path.
4. Record its checksum, access conditions, units, filters, and scoring plane.

Planned categories are reactor, spallation, CANS/HBS, diffraction, SANS, INS,
and TOF. Do not guess missing metadata and do not commit restricted data.
