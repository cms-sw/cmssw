#!/bin/bash

echo "Finding tau validation datasets for $CMSSW_VERSION"

echo "Finding QCD dataset"
QCD_DATASET=`dbs search --noheader --query="find dataset where primds=RelValQCD_FlatPt_15_3000 and release = $CMSSW_VERSION and tier=GEN-SIM-RECO"`
echo "...using $QCD_DATASET"

echo '    Scans and Deletes the old file'
let fline=`grep -n relval EventSource_QCD_RECO_cff.py | awk -F: '{print $1}'| head -n1`-1
lline=`grep -n relval EventSource_QCD_RECO_cff.py | awk -F: '{print $1}'| tail -n1`
tline=`wc -l EventSource_QCD_RECO_cff.py | awk '{print $1}'`
let ltail=tline-lline

head -n$fline EventSource_QCD_RECO_cff.py > tmp_head
tail -n$ltail EventSource_QCD_RECO_cff.py >> tmp_tail

rm EventSource_QCD_RECO_cff.py

echo "    Finding QCD files"
for file in `dbs search --noheader --query="find file where dataset=$QCD_DATASET"`; do
    echo $"   '"$file"'," >> tmp_head
done

echo "    Creating new EventSource file"
lastfile=`tail -n1 tmp_head | awk -F, '{print $1}'`
let hline=`wc -l tmp_head | awk '{print $1}'`-1
head -n$hline tmp_head > EventSource_QCD_RECO_cff.py
echo "  "$lastfile >> EventSource_QCD_RECO_cff.py
cat tmp_tail >> EventSource_QCD_RECO_cff.py

echo "    Removing temporary files"
rm tmp_head
rm tmp_tail


#-------------------------------------ZTT PART------------------------------


echo "Finding ZTT dataset"
ZTT_DATASET=`dbs search --noheader --query="find dataset where primds=RelValZTT and release = $CMSSW_VERSION and tier=GEN-SIM-RECO"`
echo "...using $ZTT_DATASET"

echo '    Scans and Deletes the old file'
let fline=`grep -n relval EventSource_ZTT_RECO_cff.py | awk -F: '{print $1}'| head -n1`-1
lline=`grep -n relval EventSource_ZTT_RECO_cff.py | awk -F: '{print $1}'| tail -n1`
tline=`wc -l EventSource_ZTT_RECO_cff.py | awk '{print $1}'`
let ltail=tline-lline

head -n$fline EventSource_ZTT_RECO_cff.py > tmp_head
tail -n$ltail EventSource_ZTT_RECO_cff.py >> tmp_tail

rm EventSource_ZTT_RECO_cff.py

echo "    Finding ZTT files"
for file in `dbs search --noheader --query="find file where dataset=$ZTT_DATASET"`; do
    echo "   '"$file"'," >> tmp_head
done

echo "    Creating new EventSource file"
lastfile=`tail -n1 tmp_head | awk -F, '{print $1}'`
let hline=`wc -l tmp_head | awk '{print $1}'`-1
head -n$hline tmp_head > EventSource_ZTT_RECO_cff.py
echo "  "$lastfile >> EventSource_ZTT_RECO_cff.py
cat tmp_tail >> EventSource_ZTT_RECO_cff.py

echo "    Removing temporary files"
rm tmp_head
rm tmp_tail
