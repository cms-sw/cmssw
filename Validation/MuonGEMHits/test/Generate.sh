### Made by Geonmo Ryu
### If you want this scripts, you need to 
### 1) pull request geonmo-cmssw/gem-sim-validation 
### 2) git cms-addpkg Validation/ and git cms-addpkg Geometry 

#!/bin/bash

cmsDriver.py SingleMuPt40Fwd_cfi -s GEN,SIM --conditions auto:run1_mc --datatier GEN-SIM --geometry Extended2023 --evt_type Validation/MuonGEMHits/SingleMuPt40Fwd_cfi --eventcontent FEVTDEBUG -n 1000 --no_exec --fileout out_sim.root --python_filename=sim_cfg.py

cmsDriver.py validation --conditions auto:run1_mc -n 1000 --eventcontent FEVTDEBUGHLT -s VALIDATION:genvalid_all --customise Validation/MuonGEMHits/gemCustum.customise2023 --datatier GEN-SIM --geometry Extended2023 --no_exec --filein file:out_sim.root --fileout file:out_valid.root --python_filename=valid_cfg.py

cmsDriver.py harvest --conditions auto:run1_mc -n 1000 --eventcontent FEVTDEBUGHLT -s HARVESTING:genHarvesting --customise Validation/MuonGEMHits/gemCustum.customise2023 --datatier GEN-SIM --geometry Extended2023 --no_exec --filein file:out_valid.root --python_filename=harvest_cfg.py
