#! /bin/csh
eval `scramv1 ru -csh`
#cmsRun runP_tracker.cfg

cmsRun trackerdigivalid.cfg 

root -b -p -q  SiPixelDigiCompare.C
root -b -p -q  SiStripDigiCompare.C
