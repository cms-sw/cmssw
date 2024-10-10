#!/bin/bash -ex

strace -z -f -o log1.txt -e trace=openat python3 -c 'import ROOT;print(ROOT.gErrorIgnoreLevel);'
if [ $(grep libFWCoreVersion.so log1.txt | wc -l) -gt 0 ] ; then
  echo "ERROR: CMSSW libraries loaded by import ROOT"
  exit 1
fi
strace -z -f -o log2.txt -e trace=openat python3 -c 'import ROOT;print(ROOT.gErrorIgnoreLevel);print(ROOT.kError);'
RUN1_OPENAT=$(grep openat log1.txt | wc -l)
RUN2_OPENAT=$(grep openat log2.txt | wc -l)
if [ ${RUN1_OPENAT} -ne  ${RUN2_OPENAT} ] ; then
  echo "ERROR: Some symbol search did not work. Two calls to ROOT loaded extra libraries."
  exit 1
fi
