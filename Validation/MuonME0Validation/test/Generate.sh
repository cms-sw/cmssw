#!/bin/bash

# VALIDATION STEP
cmsDriver.py validation --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise=Validation/MuonME0Validation/me0Custom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_local_reco_me0segment.root --fileout file:out_valid.root --python_filename=me0_valid_cfg.py

# HARVESTING STEP
cmsDriver.py harvest --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise=Validation/MuonME0Validation/me0Custom.customise2023,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixCSCAlignmentConditions --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_valid.root --python_filename=me0_harvest_cfg.py
