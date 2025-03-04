#!/bin/bash -ex

# PyROOT Test Script
#
# This script tests the following:
# 1. The ability to import and access ROOT properties (e.g., kFALSE, kTRUE).
# 2. Ensuring that PyROOT does not load any CMSSW shared libraries.
#   (Refer to: https://github.com/cms-sw/cmssw/issues/43077)

for sym in TTreeReader.fgEntryStatusText kFALSE kTRUE gStyle TTree.kMaxEntries TString.kNPOS gEnv gSystem kWarning gErrorIgnoreLevel kError ; do
  strace -z -f -o log1.txt -e trace=openat python3 -c "import ROOT;print(ROOT.${sym});"
  cmssw_lib=$(grep libFWCoreVersion.so log1.txt | wc -l)
  rm -f log1.txt
  if [ ${cmssw_lib} -gt 0 ] ; then
    echo "ERROR: CMSSW libraries loaded by import ROOT"
    exit 1
  fi
done
