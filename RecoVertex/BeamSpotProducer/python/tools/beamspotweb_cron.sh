
echo "starting:"
date
export STAGE_HOST=castorcms.cern.ch
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_6_1_patch2/src/
eval `scramv1 runtime -sh`

echo "CMSSW configured"

echo "run python script"
python /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_6_1_patch2/src/RecoVertex/BeamSpotProducer/scripts/beamvalidation.py -o /afs/cern.ch/user/y/yumiceva/www/public_html/beamspot/index.html -p /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_6_1_patch2/src/RecoVertex/BeamSpotProducer/scripts/

echo "done."
