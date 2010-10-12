#! /bin/bash

# this script is automatically called by spawnScreens.sh script for handling parallel simulations on one machine


if [ $# -ne 3 ]; then
       echo "Usage: $0 mod phase mdtau"
       echo $0 $1 $2 $3
       exit 0
fi

mod=$1
phase=$2
mdtau=$3


# TODO: adjust working dir
baseDir=/scratch/scratch0/tfruboes/2010.09.ZEmbedding_DATA/CMSSW_3_6_3/src/TauAnalysis/MCEmbeddingTools/test/pfEmbedderFullSim/Zemb_allmdtaus/


cd $baseDir
eval `scramv1 ru -sh`

#
# Adjust this - following file contains run/lumi/event list obtained using "edmFileUtil -e file:yourRootFile.root" command
#
fileEVL=ev.txt

runs=`cat $fileEVL | grep -vi lumino | awk '{ if (NF == 4)  print $1}' | sort -r | uniq`

# TODO: adjust output dir
idir=$baseDir/HLT_$mdtau/
odir=$baseDir/RECO_$mdtau/
mkdir $odir


for r in $runs ;do
  if  [ `expr $r % $mod` -ne "$phase" ]  ; then
    echo Skiping $r
    continue
  fi
  echo Running $r

  wdir=workDir_$r
  mkdir $wdir
  # prepare json
  echo '{"'$r'": [[1,999999999]]}' > $wdir/my.json
  cp embed_RECO.py $wdir
  cd $wdir

  # here we mod the copy 
  echo "process.source.fileNames = cms.untracked.vstring()" >> embed_RECO.py
  for fin in `ls $idir | grep root`; do
    echo "process.source.fileNames.extend(['file:$idir$fin'])" >> embed_RECO.py
  done


  cmsRun embed_RECO.py useJson=1 overrideBeamSpot=1
  mv embedded_RECO.root $odir/embedded_RECO_$r.root
  mv log.log $odir/log_$r
  cd ..
  ls $wdir
  rm -rf $wdir
done
