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

if [ $phase -ne 0 ]; then
  exit 0
fi


# TODO: adjust working dir
baseDir=/scratch/scratch0/tfruboes/2010.09.ZEmbedding_DATA/CMSSW_3_6_3/src/TauAnalysis/MCEmbeddingTools/test/pfEmbedderFullSim/Zemb_allmdtaus/


cd $baseDir
eval `scramv1 ru -sh`

#
# Adjust this - following file contains run/lumi/event list obtained using "edmFileUtil -e file:yourRootFile.root" command
#


# TODO: adjust output dir
idir=./RECO_$mdtau/

cmsRun merge.py dir=$idir
