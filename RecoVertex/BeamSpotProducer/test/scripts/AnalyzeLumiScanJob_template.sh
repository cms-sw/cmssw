#!/bin/sh
cd PWD
eval `scramv1 runtime -sh`
cmsRun CFGFILE >& CFGRUNLOG &

