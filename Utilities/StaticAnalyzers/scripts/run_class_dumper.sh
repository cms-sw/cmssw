#!/usr/bin/env bash
export LC_ALL=C
if [ $# -eq 0 ] ;then J=$(getconf _NPROCESSORS_ONLN); else J=$1; fi

eval `scram runtime -sh`
ulimit -m 2000000
ulimit -v 2000000
ulimit -t 1200
ulimit -f 40000000
for file in `cmsglimpse -l -F src/classes.*.h$ include`;do 
	dir=`dirname $file`;
	echo \#include \<$file\> >${LOCALRT}/src/$dir/`basename $file`.cc ; 
done
cd ${LOCALRT}/tmp/
touch dump-start
touch function-dumper.txt.unsorted plugins.txt.unsorted classes.txt.dumperct.unsorted classes.txt.dumperft.unsorted classes.txt.dumperall.unsorted
cd ${LOCALRT}/src/Utilities/StaticAnalyzers
scram b -j $J
cd ${LOCALRT}/
export USER_CXXFLAGS="-DEDM_ML_DEBUG -w"
export USER_LLVM_CHECKERS="-disable-checker cplusplus -disable-checker unix -disable-checker threadsafety -disable-checker core -disable-checker security -disable-checker deadcode -disable-checker cms -enable-checker cms.FunctionDumper -enable-checker optional.ClassDumper -enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT -enable-checker optional.EDMPluginDumper"
scram b -k -j $J checker SCRAM_IGNORE_PACKAGES=Fireworks/% SCRAM_IGNORE_SUBDIRS=test 2>&1 > $CMSSW_BASE/tmp/class+function-dumper.log
find ${LOCALRT}/src/ -name classes\*.h.cc | xargs rm -fv
cd ${LOCALRT}/tmp
touch dump-end
sort -u < plugins.txt.unsorted > plugins.txt
sort -u < classes.txt.dumperct.unsorted | grep -e"^class" >classes.txt.dumperct.sorted
awk -F\' ' {print "class \47"$2"\47\n\nclass \47"$4"\47\n\nclass \47"$6"\47\n\n" } '  <classes.txt.dumperct.sorted | sort -u >classes.txt.dumperct
sort -u < classes.txt.dumperft.unsorted | grep -e"^class" >classes.txt.dumperft.sorted
awk -F\' ' {print "class \47"$2"\47\n\nclass \47"$4"\47\n\nclass \47"$6"\47\n\n" } '  <classes.txt.dumperft.sorted | sort -u >classes.txt.dumperft
sort -u < classes.txt.dumperall.unsorted | grep -e"^class" >classes.txt.dumperall
sort -u < function-dumper.txt.unsorted > function-calls-db.txt
awk -F\' 'NR==FNR{a[$2]=1;next} { for(i in a){if($4==i && $3==" base class "){print}} }' classes.txt.dumperft.sorted classes.txt.dumperall | sort -u >classes.txt.inherits.sorted
awk -F\' 'NR==FNR{a[$2]=1;next} { for(i in a){if($2==i){ print "class \47"$2"\47\n\nclass \47"$4"\47\n\nclass \47"$6"\47\n\n" }} }' classes.txt.inherits.sorted classes.txt.dumperall | sort -u >classes.txt.inherits
cat classes.txt.dumperct classes.txt.dumperft classes.txt.inherits | grep -e"^class" | grep -v \'\' | sort -u >classes.txt
rm *.txt.*unsorted
