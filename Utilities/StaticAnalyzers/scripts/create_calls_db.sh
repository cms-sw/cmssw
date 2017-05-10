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
ulimit -m 8000000
ulimit -v 8000000
ulimit -t 1200
export USER_LLVM_CHECKERS="-disable-checker cplusplus -disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker cms.FunctionDumper -enable-checker optional.EDMPluginDumper -enable-checker cms.FunctionChecker"
cd ${LOCALRT}
touch tmp/function-dumper.txt.unsorted tmp/function-checker.txt.unsorted tmp/plugins.txt.unsorted
scram b -k -j $J checker  2>&1 > tmp/function-dumper.log
cd ${LOCALRT}/tmp
sort -u function-dumper.txt.unsorted >function-calls-db.txt
sort -u function-checker.txt.unsorted >function-statics-db.txt
sort -u plugins.txt.unsorted > plugins.txt
cat  function-calls-db.txt function-statics-db.txt >db.txt
statics.py 2>&1 >module2statics.txt.unsorted
sort -u module2statics.txt.unsorted > module2statics.txt.sorted
awk -F\' 'NR==FNR{a[" "$0"::"]=1;next} {n=0;for(i in a){if(index($2,i) && $1 == "In call stack "){n=1}}} n' plugins.txt module2statics.txt.unsorted >module2statics.txt.tmp
sort -u module2statics.txt.tmp > module2statics.txt
awk -F\' 'NR==FNR{a[" "$0"::"]=1;next} {n=0;for(i in a){if(index($4,i) && $1 == "Non-const static variable "){n=1}}} n' plugins.txt module2statics.txt.unsorted >static2modules.txt.tmp
sort -u static2modules.txt.tmp > static2modules.txt
