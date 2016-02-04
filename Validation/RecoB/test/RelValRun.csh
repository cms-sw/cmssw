#!/bin/tcsh

cd TTbar_MC
cmsRun validation_cfg.py >& mcttb.txt &
cd ../TTbar_Startup
cmsRun validation_cfg.py >& stttb.txt &
cd ../TTbar_FastSim
cmsRun validation_cfg.py >& fsttb.txt &
cd ../QCD_MC
cmsRun validation_cfg.py >& mcqcd.txt &
cd ../QCD_Startup
cmsRun validation_cfg.py >& stqcd.txt &
cd ../
