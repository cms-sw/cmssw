#! /bin/sh

numParaCompilations=1


dir=`pwd`

mdtaus=( 116 102 127 130 115 101 )
#mdtaus=( 115 )


# TODO adjust script location, pwd?
#dir=/scratch/scratch0/tfruboes/2010.09.ZEmbedding_DATA/CMSSW_3_6_3/src/TauAnalysis/MCEmbeddingTools/test/pfEmbedderFullSim/ZembRECOout/

#script=$dir/runSingle.sh
script=$dir/runSingleRECO.sh
#script=$dir/runSingleMerge.sh

for mdtau in  ${mdtaus[@]} ; do

 let phase=0
 while [ $phase -lt $numParaCompilations ]; do
   nrunning=`screen -ls | grep simu_ | wc -l ` 
   while [ $nrunning -ge $numParaCompilations  ]; do
      echo Sleeping for 1m, nrunning=$nrunning
      sleep 1m
      nrunning=`screen -ls | grep simu_ | wc -l ` 
   done

   echo nrunning=$nrunning , will  spawn $phase mdtau=$mdtau  
   screen -d -m -S simu_${phase} $script $numParaCompilations $phase $mdtau
   # $script $numParaCompilations $phase
   let phase+=1
   sleep 2s
 done

done
