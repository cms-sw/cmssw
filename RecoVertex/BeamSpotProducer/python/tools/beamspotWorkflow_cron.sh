export STAGE_HOST=castorcms.cern.ch
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch4/src/
logFileName="/afs/cern.ch/user/u/uplegger/www/Logs/MegaScriptLog.txt"
echo >> $logFileName
echo "Begin running the script on " `date` >> $logFileName
if [ ! -e .lock ]
then
  touch .lock
  eval `scramv1 runtime -sh`
  python $CMSSW_BASE/src/RecoVertex/BeamSpotProducer/scripts/BeamSpotWorkflow_T0.py -u -c BeamSpotWorkflow_T0.cfg >> $logFileName
  rm .lock
else
  echo "There is already a megascript runnning...exiting" >> $logFileName
fi
echo "Done on " `date` >> $logFileName
