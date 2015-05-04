### Made by Geonmo Ryu
### If you want this scripts, you need to 
### 1) pull request geonmo-cmssw/gem-sim-validation 
### 2) git cms-addpkg Validation/ and git cms-addpkg Geometry 

#!/bin/bash

cmsDriver.py SingleMuPt40Fwd_cfi -s GEN,SIM,DIGI,L1 --conditions auto:run1_mc --datatier GEN-SIM-DIGI --geometry Extended2023 --evt_type Validation/MuonGEMHits/SingleMuPt40Fwd_cfi --eventcontent FEVTDEBUG -n 1000 --no_exec --fileout out_digi.root --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions,SimMuon/GEMDigitizer/customizeGEMDigi.customize_digi_addGEM_muon_only --python_filename=gen_sim_digi_cfg.py

cmsDriver.py validation --conditions auto:run1_mc -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023 --datatier GEN-SIM-DIGI --geometry Extended2023 --no_exec --filein file:out_digi.root --fileout file:out_valid.root --python_filename=valid_cfg.py

cmsDriver.py harvest --conditions auto:run1_mc -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023 --datatier GEN-SIM-DIGI --geometry Extended2023 --no_exec --filein file:out_valid.root --python_filename=harvest_cfg.py
