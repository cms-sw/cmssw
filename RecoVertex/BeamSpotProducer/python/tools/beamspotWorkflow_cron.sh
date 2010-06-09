export STAGE_HOST=castorcms.cern.ch
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/user/u/uplegger/scratch0/CMSSW/CMSSW_3_6_1_patch2/src/
eval `scramv1 runtime -sh`
logFileName="/afs/cern.ch/user/u/uplegger/www/Logs/MegaScriptLog.txt"
echo >> $logFileName
echo "Begin running the script on " `date` >> $logFileName
python $CMSSW_BASE/src/RecoVertex/BeamSpotProducer/scripts/BeamSpotWorkflow_T0.py -u -l lockBeamSpotWorkflow_T0 -c BeamSpotWorkflow_T0.cfg >> $logFileName
echo "Done on " `date` >> $logFileName
