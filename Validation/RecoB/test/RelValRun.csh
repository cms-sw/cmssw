#!/bin/tcsh

cd TTbar_Startup
cmsRun reco_validation_cfg.py >& stttb.txt &
cd ../TTbar_Startup_PU
cmsRun reco_validation_cfg.py >& puttb.txt &
cd ../TTbar_FastSim
cmsRun reco_validation_cfg.py >& fsttb.txt &
cd ../QCD_Startup
cmsRun reco_validation_cfg.py >& stqcd.txt &
cd ../DATA
cmsRun reco_validation_cfg.py >& data.txt &
cd ../
