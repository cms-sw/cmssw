#! /bin/csh


set scriptDir = `pwd`
set script = testRPCTriggerEff.py
set loc = 'https://cmsweb.cern.ch/dbs_discovery/getLFNsForSite?dbsInst=cms_dbs_prod_global&site=cmssrm.fnal.gov&datasetPath=/RelValSingleMuPt100/CMSSW_3_0_0_pre7_IDEAL_30X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG&what=txt&userMode=expert&run='


set files = `wget -o /dev/null --no-check-certificate -O - "$loc"`

mkdir out
rm -rf out/*
@ i = 0
foreach file ($files)

  bsub runsingle.csh $i $scriptDir $script $file
  @ i += 1

end
