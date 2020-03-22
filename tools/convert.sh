#!/usr/bin/env bash

for file in $(find src -type f -name '*.py')
do
  if grep -q "# %%" "$file"; then
    jupytext --to notebook $file
  fi
done
