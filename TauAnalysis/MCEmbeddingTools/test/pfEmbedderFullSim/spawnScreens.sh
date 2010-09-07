#! /bin/sh

numParaCompilations=5

let phase=0

dir=`pwd`

# TODO adjust script location
script=/scratch/cms32/tfruboes/2010.07.RunEmbeddingOnData/testMerge/CMSSW_3_6_1_patch4/src/TauAnalysis/MCEmbeddingTools/test/pfEmbedderFullSim/runSingle.sh


while [ $phase -lt $numParaCompilations ]; do
  echo Spawn $phase
  screen -d -m -S simu_${phase} $script $numParaCompilations $phase
  let phase+=1
done
