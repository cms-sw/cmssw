#! /bin/csh
eval `scramv1 runtime -csh`

cmsRun SiStripRecHitsValid_cfg.py

root -b -p -q SiStripRecHitsCompare.C
