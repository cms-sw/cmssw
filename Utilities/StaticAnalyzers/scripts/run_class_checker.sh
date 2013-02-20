#!/usr/bin/env bash
export SCRAM_ARCH=slc5_amd64_gcc472
cd ${CMSSW_BASE}
eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 3600
for file in `cmsglimpse -l -F src/classes.h$ include`;do dir=`dirname $file`;echo \#include \<$file\> >${CMSSW_BASE}/src/$dir/classes_def.cc ; done
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT"
scram b -k -j $1 checker 2>&1 | tee /tmp/classdumper.log
mv /tmp/classes.txt /tmp/classes.txt.unsorted
sort /tmp/classes.txt.unsorted >/tmp/classes.txt
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassChecker"
scram b -k -j $1 checker 2>&1 | tee /tmp/classchecker.log
