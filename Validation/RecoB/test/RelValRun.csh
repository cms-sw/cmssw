#!/bin/tcsh

cd TTbar_Startup
cmsRun validation_cfg.py >& stttb.txt &
cd ../TTbar_Startup_PU
cmsRun validation_cfg.py >& puttb.txt &
cd ../TTbar_FastSim
cmsRun validation_cfg.py >& fsttb.txt &
cd ../QCD_Startup
cmsRun validation_cfg.py >& stqcd.txt &
cd ../
