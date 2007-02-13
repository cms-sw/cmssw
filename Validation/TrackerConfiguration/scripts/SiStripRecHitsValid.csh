#! /bin/csh
eval `scramv1 runtime -csh`

cmsRun SiStripRecHitsValid.cfg

root -b -p -q SiStripRecHitsCompare.C
