#!/usr/bin/env bash

for file in $(find src -type f -name '*.py')
do
  # Convert scripts containing notebook syntax.
  if grep -q "# %%" "$file"; then
    out=$(echo $file | sed -e "s/\.py/\.ipynb/g" | sed -e "s/src/src\/notebooks/g")
    jupytext --to notebook $file -o $out
  fi
done
