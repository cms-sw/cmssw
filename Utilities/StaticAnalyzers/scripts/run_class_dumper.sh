#!/usr/bin/env bash
if [ $# -eq 0 ]
        then
        echo "Passing -j1 to make."
        echo "Supply a number argument to override."
        J=1
else
        J=$1
fi

export SCRAM_ARCH=slc5_amd64_gcc472
eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 3600
for file in `cmsglimpse -l -F src/classes.h$ include`;do dir=`dirname $file`;echo \#include \<$file\> >${CMSSW_BASE}/src/$dir/classes_def.cc ; done
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassDumperCT "
scram b -k -j $J checker 
mv ${CMSSW_BASE}/tmp/classes.txt.unsorted ${CMSSW_BASE}/tmp/classes.txt.dumperct.unsorted
sort -u ${CMSSW_BASE}/tmp/classes.txt.dumperct.unsorted |grep -v -e"^class std::">${CMSSW_BASE}/tmp/classes.txt.dumperct
find ${CMSSW_BASE}/src/ -name classes_def.cc | xargs rm -f
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassDumperFT"
scram b -k -j $J checker 
mv ${CMSSW_BASE}/tmp/classes.txt.unsorted ${CMSSW_BASE}/tmp/classes.txt.dumperft.unsorted
sort -u ${CMSSW_BASE}/tmp/classes.txt.dumperft.unsorted |grep -v -e"^class std::">${CMSSW_BASE}/tmp/classes.txt.dumperft
export USER_LLVM_CHECKERS="-disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassDumperInherit"
scram b -k -j $J checker 
mv ${CMSSW_BASE}/tmp/classes.txt.unsorted ${CMSSW_BASE}/tmp/classes.txt.inherits.unsorted
sort -u ${CMSSW_BASE}/tmp/classes.txt.inherits.unsorted |grep -v -e"^class std::">${CMSSW_BASE}/tmp/classes.txt.inherits
cat ${CMSSW_BASE}/tmp/classes.txt.dumperct ${CMSSW_BASE}/tmp/classes.txt.dumperft ${CMSSW_BASE}/tmp/classes.txt.inherits | sort -u >${CMSSW_BASE}/tmp/classes.txt
