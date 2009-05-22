#! /bin/csh


set scriptDir = `pwd`
set script = testRPCTriggerEff.py
set dsets = "/RelValSingleMuPt10/CMSSW_3_1_0_pre7_IDEAL_31X_v1/GEN-SIM-RECO \
             /RelValSingleMuPt100/CMSSW_3_1_0_pre7_IDEAL_31X_v1/GEN-SIM-RECO \
             /RelValSingleMuPt1000/CMSSW_3_1_0_pre7_IDEAL_31X_v1/GEN-SIM-RECO" 

mkdir out
rm -rf out/*

@ i = 0
foreach d ($dsets) 
  if ($d != "") then

      set loc = 'https://cmsweb.cern.ch/dbs_discovery/getLFNsForSite?dbsInst=cms_dbs_prod_global&site=cmssrm.fnal.gov&datasetPath='"$d"'&what=txt&userMode=expert&run='
      set files = `wget -o /dev/null --no-check-certificate -O - "$loc"`
      foreach file ($files)
        bsub runsingle.csh $i $scriptDir $script $file
        @ i += 1
      end

 endif
end
exit


