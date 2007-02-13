#! /bin/csh
eval `scramv1 runtime -csh`

cmsRun SiPixelRecHitsValid.cfg

root -b -p -q SiPixelRecHitsCompare.C

