### Made by Geonmo Ryu
### If you want this scripts, you need to 
### 1) pull request geonmo-cmssw/gem-sim-validation 
### 2) git cms-addpkg Validation/ and git cms-addpkg Geometry 

#!/bin/bash




## For GE21 v7 20deg
cmsDriver.py SingleMuPt100_pythia8_cfi -s GEN,SIM,DIGI,L1 --conditions auto:run2_design --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --eventcontent FEVTDEBUGHLT  --customise=SimMuon/GEMDigitizer/customizeGEMDigi.customize_digi_addGEM_muon_only,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions,SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7,SLHCUpgradeSimulations/Configuration/me0Customs.customise_Digi -n 100 --no_exec --fileout out_digi.root --python_filename SingleMuPt100_cfi_GEM-SIM-DIGI_Extended2015_GE21v7_cfg.py --era Run2_25ns

cmsDriver.py validation --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_local_reco.root --fileout file:out_valid.root --python_filename=valid_GE21v7_cfg.py --era Run2_25ns

cmsDriver.py harvest --conditions auto:run2_design -n -1 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_valid.root --python_filename=harvest_GE21_v7_cfg.py --era Run2_25ns

## For GE21 v7 10deg
cmsDriver.py SingleMuPt100_pythia8_cfi -s GEN,SIM,DIGI,L1 --conditions auto:run2_design --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --eventcontent FEVTDEBUGHLT  --customise=SimMuon/GEMDigitizer/customizeGEMDigi.customize_digi_addGEM_muon_only,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions,SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7_10deg,SLHCUpgradeSimulations/Configuration/me0Customs.customise_Digi -n 100 --no_exec --fileout out_digi.root --python_filename SingleMuPt100_cfi_GEM-SIM-DIGI_Extended2015_GE21v7_10deg_cfg.py --era Run2_25ns

cmsDriver.py validation --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7_10deg --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_local_reco.root --fileout file:out_valid.root --python_filename=valid_GE21v7_10deg_cfg.py --era Run2_25ns

cmsDriver.py harvest --conditions auto:run2_design -n -1 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise SLHCUpgradeSimulations/Configuration/gemCustom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE21_v7_10deg --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_valid.root --python_filename=harvest_GE21_v7_10deg_cfg.py --era Run2_25ns
