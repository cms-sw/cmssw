#!/bin/bash
FILES="$@"
for i in $FILES
do
echo "Prcoessing image $i ..."
convert -thumbnail x200 $i thumb.$i.jpg
done
