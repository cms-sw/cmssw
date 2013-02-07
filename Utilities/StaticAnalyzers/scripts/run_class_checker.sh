#!/usr/bin/env bash
eval `scram runtime -sh`
for file in `cmsglimpse -l -F src/classes.h$ include`;do dir=`dirname $file`;echo \#include \<$file\> >${CMSSW_BASE}/src/$dir/classes_def.cc ; done
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT"
ulimit -m 2000000
ulimit -v 2000000
scram b -j $1 checker 2>&1 | tee /tmp/classdumper.log
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassChecker"
scram b -j $1 checker 2>&1 | tee /tmp/classchecker.log
