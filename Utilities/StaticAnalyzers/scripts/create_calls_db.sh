#!/usr/bin/env bash
if [ $# -eq 0 ]
        then
        echo "Passing -j1 to make."
        echo "Supply a number argument to override."
        J=1
else
        J=$1
fi

eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 1200
export USER_LLVM_CHECKERS="-disable-checker cplusplus -disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker cms.FunctionDumper -enable-checker optional.EDMPluginDumper -enable-checker cms.FunctionChecker"
cd ${CMSSW_BASE}
scram b -k -j $J checker  2>&1 > tmp/function-dumper.log
cd ${CMSSW_BASE}/tmp
sort -u function-dumper.txt.unsorted >function-calls-db.txt
sort -u function-checker.txt.unsorted >function-statics-db.txt
sort -u plugins.txt.unsorted >function-plugins-db.txt
cat  function-calls-db.txt function-statics-db.txt function-plugins-db.txt >db.txt
statics.py 2>&1 >module2statics.txt.unsorted
sort -u module2statics.txt.unsorted >module2statics.txt
