#!/bin/tcsh

#Check for correct number of arguments
if ($#argv<4) then
    echo "Script needs 4 arguments for comparison:"
    echo "validation label, reference label, validation file and reference file"
    exit
endif

#Check to see if the CMS environment is set up
if ($?CMSSW_BASE != 1) then
    echo "CMS environment not set up"
    exit
endif

set val_lab=$1
set ref_lab=$2
set val_file=$3
set ref_file=$4
set cur_dir=`pwd`
set finaldir=$CMSSW_VERSION

mkdir ${val_lab}_vs_${ref_lab}


cat ValidationBTag_OneSampleTemplate.xml | sed -e s%CURDIR%${cur_dir}%g | sed -e s/REF_LABEL/${ref_lab}/g | sed -e s/VAL_LABEL/${val_lab}/g | sed -e s/REF_FILE/${ref_file}/g | sed -e s/VAL_FILE/${val_file}/g > ${val_lab}_vs_${ref_lab}/ValidationBTag_catprod.xml

cd ${val_lab}_vs_${ref_lab}
cuy.py -b -x ValidationBTag_catprod.xml -p gif << +EOF
q
+EOF

cd ../
