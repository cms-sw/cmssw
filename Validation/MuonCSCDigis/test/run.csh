#!/bin/csh -v

eval `scramv1 ru -csh`
set suffix = $1
sed "s/FOO/$suffix/g" cscDigiValidation.cfg >! cscDigiValidation{$suffix}.cfg

cmsRun  cscDigiValidation{$suffix}.cfg

#root -b -q CSCOval.C\(\"{$suffix}\",\"gif\"\)

