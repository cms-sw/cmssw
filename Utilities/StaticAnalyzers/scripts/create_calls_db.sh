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

eval `scram runtime -sh`
ulimit -m 8000000
ulimit -v 8000000
ulimit -t 1200
export USER_LLVM_CHECKERS="-disable-checker cplusplus -disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker cms.FunctionDumper -enable-checker optional.EDMPluginDumper -enable-checker cms.FunctionChecker -enable-checker optional.ClassDumper -enable-checker optional.ClassChecker"
export USER_CXXFLAGS="-DEDM_ML_DEBUG"
cd ${LOCALRT}/src/Utilities/StaticAnalyzers
scram b clean; scram b -k -j $J 2>&1 | tee ${LOCALRT}/tmp/clangSAbuild.log
cd ${LOCALRT}/tmp
if [ ! -f ./classes.txt ] 
        then 
        cp  -p ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/classes.txt* . 
fi
touch function-dumper.txt.unsorted function-checker.txt.unsorted plugins.txt.unsorted class-checker.txt.unsorted
cd ${LOCALRT}
scram b -k -j $J checker  SCRAM_IGNORE_PACKAGES=Fireworks/% SCRAM_IGNORE_SUBDIRS=test 2>&1 > tmp/function+class-dumper.log
cd ${LOCALRT}/tmp

sort -u < function-dumper.txt.unsorted >function-calls-db.txt
sort -u < function-checker.txt.unsorted >function-statics-db.txt
sort -u < plugins.txt.unsorted > plugins.txt
cat  function-calls-db.txt function-statics-db.txt >db.txt
sort -u < classes.txt.dumperall.unsorted >classes.txt.dumperall
sort -u < class-checker.txt.unsorted | grep -e"^data class">class-checker.txt

