#! /bin/csh
eval `scramv1 ru -csh`

cmsRun trackerdigivalid_cfg.py

root -b -p -q  SiPixelDigiCompare.C
root -b -p -q  SiStripDigiCompare.C

