
echo "starting:"
date
export STAGE_HOST=castorcms.cern.ch
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_5_6/src/
eval `scramv1 runtime -sh`

echo "CMSSW configured"

python /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_5_6/src/RecoVertex/BeamSpotProducer/test/tools/beamspotweb_cron.py

