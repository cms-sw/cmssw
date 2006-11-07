#!/bin/sh

sed "s|VALUE1|${3}|"    ../templates/template-${1}Source-${2}.cff | \
sed "s|VALUE2|${4}|" >  ../source/${1}Source-${2}_${3}_${4}.cff

sed "s|_PROCESS_|${2}|" ../templates/template-${1}SimDigi.cfg     | \
sed "s|_MIN_|_${3}|"                                         | \
sed "s|_MAX_|_${4}|" >  ../${2}_${3}_${4}.cfg

echo "Created .cfg File for ${2}_${3}_${4}"
