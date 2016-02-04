#! /bin/csh
eval `scramv1 runtime -csh`

cmsRun SiPixelRecHitsValid_cfg.py

root -b -p -q SiPixelRecHitsCompare.C

