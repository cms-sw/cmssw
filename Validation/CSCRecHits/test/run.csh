#!/bin/csh -v

eval `scramv1 ru -csh`
set suffix = $1
sed "s/ref/$suffix/g" cscRecHitValidation.cfg >! cscRecHitValidation{$suffix}.cfg

cmsRun  cscRecHitValidation{$suffix}.cfg

#root -b -q CSCOval.C\(\"{$suffix}\",\"gif\"\)

