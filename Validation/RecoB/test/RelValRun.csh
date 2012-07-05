#!/bin/tcsh

cd TTbar_Startup
cmsRun validation_cfg.py jets="ak5PFJEC" >& stttb.txt &
cd ../TTbar_Startup_PU
cmsRun validation_cfg.py jets="ak5PFnoPU" >& puttb.txt &
cd ../TTbar_FastSim
cmsRun validation_cfg.py jets="ak5PFJEC" >& fsttb.txt &
cd ../QCD_Startup
cmsRun validation_cfg.py jets="ak5PFJEC" >& stqcd.txt &
cd ../
