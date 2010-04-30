
echo "starting:"
date
export STAGE_HOST=castorcms.cern.ch
source /afs/cern.ch/cms/sw/cmsset_default.sh
cd /afs/cern.ch/user/y/yumiceva/scratch0/CMSSW_3_5_7/src/
eval `scramv1 runtime -sh`

echo "CMSSW configured"
cd /afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/yumiceva/tmp_lumi_workflow/

echo "run python script"
python /afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/yumiceva/tmp_lumi_workflow/runonexpress.py

echo "done."
