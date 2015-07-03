#!/bin/bash

# GEN-SIM-DIGI STEP
cmsDriver.py SingleMuPt100_cfi -s GEN,SIM,DIGI --conditions auto:run2_design --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --eventcontent FEVTDEBUGHLT --era Run2_25ns --customise=SLHCUpgradeSimulations/Configuration/gemCustoms.customise2023,SLHCUpgradeSimulations/Configuration/me0Customs.customise,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions -n 1000 --no_exec --fileout out_digi.root --python_filename SingleMuPt100_cfi_GEM-SIM-DIGI_Extended2015MuonGEMDev_cfg.py

# VALIDATION STEP
cmsDriver.py validation --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise=Validation/MuonME0Hits/me0Custom.customise2023 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_digi.root --fileout file:out_valid.root --python_filename=me0_valid_cfg.py

# HARVESTING STEP
cmsDriver.py harvest --conditions auto:run2_design -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise=Validation/MuonME0Hits/me0Custom.customise2023 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --no_exec --filein file:out_valid.root --python_filename=me0_harvest_cfg.py
