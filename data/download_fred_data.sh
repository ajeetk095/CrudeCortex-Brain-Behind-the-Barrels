#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/fred
for sid in WTISPLC INDPRO CPIENGSL DTWEXBGS VIXCLS; do
  curl -fL "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${sid}" \
    -o "data/fred/${sid}.csv"
done

echo "Downloaded FRED CSV files to data/fred/"
