#!/bin/sh
#
# 05/05/2010 - AC
#

##========================================================================
## Checking the script input parameters
##========================================================================

if [ [$1 -a $2] -a $3 ]; then
echo -e "---**----------------------------------------------------------------------------------------------" 
else 
echo -e "---**--- Please use the script in this way : " 
echo -e "---**--- ./theScript.sh [-e | -n ] [ MinBias | MinBiasRaw | ZeroBias | ZeroBiasRaw | TestEnables | HLTMON | ExpressPhysics| GOODCOLL ] <run>" 
exit
fi 


##========================================================================
## Defining the log file name
##========================================================================


if [ "$1" == "-n" ]; then 
 DataType="Ntuple"
fi 
if [ "$1" == "-e" ]; then
 DataType="Emulator"
fi

timeStamp=`date +%d-%m-%Y_%Hh%Mmn%Ss`
pwd=`pwd`
log=${pwd}/$2_${DataType}_$3_${timeStamp}.log




##========================================================================
## Defining castor output directories
##========================================================================
castorUserDir=/castor/cern.ch/cms/store/caf/user/$USER/L1PromptAnalysis_$2_${DataType}_$3
castorL1Dir=/castor/cern.ch/cms/store/caf/user/L1AnalysisNtuples/



##========================================================================
echo -e "---**--- Setting the environnement..." | tee -a $log
echo -e "---**--- Sourcing : /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/l1analysis/scripts/setup.sh\n" | tee -a $log
##========================================================================
#source /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/l1analysis/scripts/setup.sh
source /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/alebihan/CMSSW_3_5_7/src/UserCode/L1TriggerDPG/scripts/setup.sh


##========================================================================
echo -e "---**---------------------------------------------------------------------------------------------"| tee -a $log
echo -e "---**--- Submitting prompt analysis to crab..." | tee -a $log
##========================================================================


#
# Checking if the dataset has already been processed : check local, l1 and castor areas...
#
#if [ -r /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/l1analysis/ntuples/$2_${DataType}_$3.root ]; then
#  echo -e "---**--- Dataset has already been processed, file exists there :  " | tee -a $log
#  ls -rtl /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/l1analysis/ntuples/$2_${DataType}_$3.root | tee -a $log
#  exit
#fi
if [ -d $2_${DataType}_$3 ]; then
  echo -e "---**--- Directory "$2_${DataType}_$3 "already exists, crab won't be running, exit." | tee -a $log
  echo -e "---**--- Please do : rm -r "$2_${DataType}_$3 | tee -a $log
  exit
fi
echo -e "---**--- Checking if the user castor output directory "$castorUserDir" is empty..." | tee -a $log
rfdir $castorUserDir
if [ $? = 0 ]; then
   echo -e "---**--- Castor output directory already exists, crab won't be running, exit." | tee -a $log
   echo -e "---**--- Please do : rfrm -r " $castorUserDir | tee -a $log
   exit
else
echo -e "---**--- ok, castor output directory is empty !"
fi


#
#launch crab...
#
echo -e "\n---**--- The produced data will be "$DataType"s\n" | tee -a $log
submit.py $1 $2 $3 | tee -a $log
echo -e "\n---**--- Waiting 10 mn, until the crab submission is done..." | tee -a $log
sleep 600


  
 
##========================================================================
echo -e "\n\n---**----------------------------------------------------------------------------------------------" | tee -a $log
echo -e "---**--- Checking the current crab status..." | tee -a $log
##========================================================================


nRetrieved=`crab -status -c $2_${DataType}_$3 | grep 'Retrieved' | wc -l`
nCreated=`crab -status -c $2_${DataType}_$3 | grep 'Created' | wc -l`

if [ $nRetrieved -a {$nRetrieved = $nTotal} ]; then
    echo -e "---**--- CRAB jobs have already been retrieved, exit !" | tee -a $log
    exit
fi

if [ $nCreated -a {$nCreated = $nTotal} ]; then
    echo -e "---**--- CRAB jobs are in created status, problem during submission, exit !" | tee -a $log
    exit
fi
 
#
# start an endless loop to check the crab status each 10 mn 
# exit the loop once crab is done 
#
 
 
while [ 1 = 1 ]
do
   nTotal=`crab -status -c $2_${DataType}_$3 | grep 'Total' | awk '{print $2}'`
   nDone=`crab -status -c $2_${DataType}_$3 | grep 'Done' | wc -l`
    
   echo -e "---**--- TOTAL jobs : " $nTotal  | tee -a $log
   echo -e "---**--- DONE jobs : " $nDone    | tee -a $log
   
   if [ $nDone = $nTotal ]; then
      echo -e "---**--- CRAB IS DONE, all jobs have been processed !"  | tee -a $log
      break
   fi
   
   if [ "$nTotal" == "" ]; then
      echo -e "---**--- No access to crab status, exit !"  | tee -a $log
      exit
   fi

   echo -e "---**--- Checking again in 10 mn...\n"  | tee -a $log
   sleep 600
done




##========================================================================
echo -e "\n\n\n---**----------------------------------------------------------------------------------------------" | tee -a $log
echo -e "---**---  Harvesting the prompt analysis root files stored on :" | tee -a $log
echo -e "---**--- "$castorUserDir | tee -a $log
##========================================================================
rfdir  $castorUserDir 2>&1 | tee -a $log
harvest.py $2_${DataType}_$3 | tee -a $log

echo -e "\n---**--- Checking the harvested file :" | tee -a $log
rfdir $castorL1Dir/$2_${DataType}_$3.root 2>&1 | tee -a  $log
if [ "rfdir $castorL1Dir/$2_${DataType}_$3.root 2>&1 | awk  '{print $2}'" = "No" ]; then
   echo -e "exit !" | tee -a $log
   exit
fi




##========================================================================
echo -e "\n---**----------------------------------------------------------------------------------------------" | tee -a $log
echo -e "---**--- Analyzing the prompt analysis root files..." | tee -a $log
##======================================================================== 

cd /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER/l1analysis/cmssw/CMSSW_3_5_7_beta/src/ 
eval `scram ru -sh`		  
cd UserCode/L1TriggerDPG/macros/ 

#
# make sure no other root output is being analyzed at the same time...
# to be modified...

if [ -f analysisOngoing ]; then
  echo -e "Root analysis already ongoing, exit ! " | tee -a $log
  exit
else
  touch analysisOngoing
  rm L1Tree.root   2>&1  | tee -a  $log
  rfcp $castorL1Dir/$2_${DataType}_$3.root L1Tree.root   2>&1  | tee -a  $log
  if [ -f L1Tree.root ]; then
     root -q initL1Analysis.C | tee -a $log
     rm analysisOngoing 2>&1  | tee -a  $log
  else 
     echo -e "Problem copying file from castor, exit !" | tee -a $log
     rm analysisOngoing 2>&1  | tee -a  $log
     exit  
  fi   
fi


#rm -rf plotDir     2>&1  | tee -a  $log
#mkdir -pv plotDir  2>&1  | tee -a  $log
#mv *.jpeg plotDir  2>&1  | tee -a  $log
#cp diow.pl plotDir 2>&1  | tee -a  $log
#cd plotDir			 
#./diow.pl		  | tee -a  $log 
