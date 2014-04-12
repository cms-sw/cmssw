#!/usr/bin/env bash
export LC_ALL=C 
if [ $# -eq 0 ] 
 	then
	echo "Passing -j1 to make."
	echo "Supply a number argument to override."
	J=1
else
	J=$1
fi

export SCRAM_ARCH=slc5_amd64_gcc481
eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 1200
export USER_LLVM_CHECKERS="-disable-checker cplusplus -disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker optional.ClassChecker"
if [ ! -f ${CMSSW_BASE}/tmp/classes.txt ] 
	then 
	cp  -p ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/classes.txt ${CMSSW_BASE}/tmp/classes.txt
fi
mv ${CMSSW_BASE}/tmp/class-checker.txt.sorted ${CMSSW_BASE}/tmp/class-checker.txt.sorted.old
rm ${CMSSW_BASE}/tmp/class-checker.txt.unsorted
scram b -k -j $J checker 2>&1 | tee ${CMSSW_BASE}/tmp/classchecker.log
sort -u < ${CMSSW_BASE}/tmp/class-checker.txt.unsorted | grep -e"^data class">${CMSSW_BASE}/tmp/class-checker.txt.sorted 
