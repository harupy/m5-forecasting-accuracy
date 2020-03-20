#!/usr/bin/env bash

slug="m5-forecasting-accuracy"
zip_name="${slug}.zip"
out_dir="input/${slug}"

# Make the output director if it does not exists.
if [ -d "$out_dir" ]; then
  mkdir -p $out_dir
fi

kaggle competitions download -c $slug
unzip -o $zip_name -d $out_dir
rm $zip_name
