#!/usr/bin/env bash
export LC_ALL=C
if [ $# -eq 0 ] ;then J=$(getconf _NPROCESSORS_ONLN); else J=$1; fi

eval `scram runtime -sh`
for file in `cmsglimpse -l -F src/classes.*.h$ include | sed -e 's|^src/||'`;do
     dir=`dirname $file`;
     echo \#include \<$file\> >${LOCALRT}/src/$dir/`basename $file`.cc ; 
done
cd ${LOCALRT}/tmp/
touch dump-start
#touch function-dumper.txt.unsorted plugins.txt.unsorted classes.txt.dumperct.unsorted classes.txt.dumperft.unsorted classes.txt.dumperall.unsorted
cd ${LOCALRT}/src/Utilities/StaticAnalyzers
scram b -j $J
cd ${LOCALRT}/
export USER_CXXFLAGS="-DEDM_ML_DEBUG -w"
export USER_LLVM_CHECKERS="-enable-checker cms.FunctionDumper -enable-checker optional.ClassDumper -enable-checker optional.ClassDumperCT -enable-checker optional.ClassDumperFT -enable-checker optional.EDMPluginDumper"
scram b -k -j $J checker SCRAM_IGNORE_PACKAGES=Fireworks/% SCRAM_IGNORE_SUBDIRS=test > $LOCALRT/tmp/class+function-dumper.log 2>&1
find ${LOCALRT}/src/ -name classes\*.h.cc | xargs rm -fv
cd ${LOCALRT}/tmp
touch dump-end
sort -u < plugins.txt.unsorted > plugins.txt
sort -u < classes.txt.dumperct.unsorted | grep -e"^class" >classes.txt.dumperct.sorted
sort -u < classes.txt.dumperct.unsorted | grep -v -e"^class" >classes.txt.dumperct.extra
awk -F\' ' {print "class \47"$2"\47\n\nclass \47"$4"\47\n\nclass \47"$6"\47\n\n" } '  <classes.txt.dumperct.sorted | sort -u >classes.txt.dumperct
sort -u < classes.txt.dumperft.unsorted | grep -e"^class" >classes.txt.dumperft.sorted
sort -u < classes.txt.dumperft.unsorted | grep -v -e"^class" >classes.txt.dumperft.extra
awk -F\' ' {print "class \47"$2"\47\n\nclass \47"$4"\47\n\nclass \47"$6"\47\n\n" } '  <classes.txt.dumperft.sorted | sort -u >classes.txt.dumperft
sort -u < classes.txt.dumperall.unsorted | grep -e"^class" >classes.txt.dumperall
sort -u < classes.txt.dumperall.unsorted | grep -v -e"^class" >classes.txt.dumperall.extra
sort -u < function-dumper.txt.unsorted > function-calls-db.txt
class-composition.py >classes.txt.inherits.unsorted
sort -u classes.txt.inherits.unsorted | grep -e"^class" | grep -v \'\' >classes.txt.inherits
sort -u classes.txt.inherits.unsorted | grep -v -e"^class" >classes.txt.inherits.extra
sort -u getparam-dumper.txt.unsorted | awk '{print $0"\n"}' >getparam-dumper.txt
cat classes.txt.inherits classes.txt.dumperft classes.txt.dumperct | grep -e"^class" | grep -v \'\' | sort -u >classes.txt
rm *.txt.*unsorted
classnames-extract.py
bloom_filter_generator bloom.bin classnames.txt
cp -pv $LOCALRT/tmp/bloom.bin $LOCALRT/src/Utilities/StaticAnalyzers/scripts/bloom.bin
