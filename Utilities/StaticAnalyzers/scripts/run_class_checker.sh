#!/usr/bin/env bash
export SCRAM_ARCH=slc5_amd64_gcc472
cd ${CMSSW_BASE}
eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 3600
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassChecker"
scram b -k -j $1 checker 2>&1 | tee /tmp/classchecker.log
