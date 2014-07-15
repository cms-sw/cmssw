### Made by Geonmo Ryu
### If you want this scripts, you need to 
### 1) pull request geonmo-cmssw/gem-sim-validation 
### 2) git cms-addpkg Validation/ and git cms-addpkg Geometry 

#!/bin/bash
cmsDriver.py SingleMuPt100_cfi -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --conditions auto:upgradePLS3 --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v1 --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2023Muon --eventcontent FEVTDEBUGHLT -n 1000 --no_exec --fileout file:out_reco_v6.root --python_filename=reco_v6_cfg.py
cmsDriver.py SingleMuPt100_cfi -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO --conditions auto:upgradePLS3 --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v2 --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2023Muon --eventcontent FEVTDEBUGHLT -n 1000 --no_exec --fileout file:out_reco_v7.root --python_filename=reco_v7_cfg.py

cmsDriver.py validation --conditions auto:upgradePLS3 -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --datatier GEN-SIM-RECO --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v1 --geometry Extended2023Muon --magField 38T_PostLS1 --no_exec --filein file:out_reco_v6.root --fileout file:out_valid_v6.root --python_filename=valid_v6_cfg.py
cmsDriver.py validation --conditions auto:upgradePLS3 -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --datatier GEN-SIM-RECO --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v2 --geometry Extended2023Muon --magField 38T_PostLS1 --no_exec --filein file:out_reco_v7.root --fileout file:out_valid_v7.root --python_filename=valid_v7_cfg.py

cmsDriver.py harvest --conditions auto:upgradePLS3 -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --datatier GEN-SIM-DIGI --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v1 --geometry Extended2023Muon --magField 38T_PostLS1 --no_exec --filein file:out_valid_v6.root --python_filename=harvest_v6_cfg.py

cmsDriver.py harvest --conditions auto:upgradePLS3 -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --datatier GEN-SIM-DIGI --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon,Geometry/GEMGeometry/gemGeometryCustoms.custom_GE11_8and8partitions_v2 --geometry Extended2023Muon --magField 38T_PostLS1 --no_exec --filein file:out_valid_v7.root --python_filename=harvest_v7_cfg.py
