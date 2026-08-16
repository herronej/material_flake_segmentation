#!/bin/bash
# Reassemble the split archives from the Hugging Face repo and unpack them.
#
#   huggingface-cli download openhcsanyu/2DFlakeSemSeg --repo-type dataset --local-dir raw
#   bash scripts/prepare_data.sh raw data
#
# The archives are split zips (graphene.zip.001..004, MoS2.zip.001..002). Do not
# rename the parts. `7z x` on the first part pulls in the rest automatically.

set -euo pipefail
RAW="${1:-raw}"
OUT="${2:-data}"

command -v 7z >/dev/null || { echo "7z not found (apt install p7zip-full)"; exit 1; }
mkdir -p "$OUT"

for material in graphene MoS2; do
  first="$RAW/${material}.zip.001"
  [ -f "$first" ] || { echo "missing $first, skipping $material"; continue; }
  echo "extracting $material ..."
  7z x -y -o"$OUT" "$first" >/dev/null
done

echo
echo "Layout check:"
for d in "$OUT"/*/; do
  name=$(basename "$d")
  for split in train2024 val2024; do
    imgs=$(find "$d/$split" -maxdepth 1 -type f 2>/dev/null | wc -l)
    msks=$(find "$d/annotations_semseg/$split" -maxdepth 1 -type f 2>/dev/null | wc -l)
    printf "  %-12s %-10s images=%-6s masks=%s\n" "$name" "$split" "$imgs" "$msks"
    [ "$imgs" = "$msks" ] || echo "    WARNING: image/mask count mismatch"
  done
done

echo
echo "Next: python scripts/inspect_data.py --root $OUT/graphene --split train2024 --limit 50"
