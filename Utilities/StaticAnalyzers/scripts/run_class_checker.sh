#!/usr/bin/env bash
export LC_ALL=C 
if [ $# -eq 0 ] ;then J=$(getconf _NPROCESSORS_ONLN); else J=$1; fi

eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 1200
ulimit -f 40000000
if [ ! -f ${LOCALRT}/tmp/classes.txt ] 
	then 
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_dumper.sh first"
	exit 1
fi
cd ${LOCALRT}/tmp/
touch check-start
touch function-checker.txt.unsorted class-checker.txt.unsorted
cd ${LOCALRT}/src/Utilities/StaticAnalyzers
scram b -j $J
cd ${LOCALRT}/
export USER_CXXFLAGS="-DEDM_ML_DEBUG -w"
export USER_LLVM_CHECKERS="-enable-checker threadsafety -enable-checker optional.ClassChecker -enable-checker cms.FunctionChecker"
scram b -k -j $J checker  SCRAM_IGNORE_PACKAGES=Fireworks/% SCRAM_IGNORE_SUBDIRS=test 2>&1 > ${LOCALRT}/tmp/class+function-checker.log
cd ${LOCALRT}/tmp/
touch check-end
sort -u < class-checker.txt.unsorted | grep -e"^data class">class-checker.txt
sort -u < function-checker.txt.unsorted >function-statics-db.txt
rm *.txt.unsorted
