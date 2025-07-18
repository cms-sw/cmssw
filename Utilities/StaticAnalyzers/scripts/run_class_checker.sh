#!/usr/bin/env bash
export LC_ALL=C 
if [ $# -eq 0 ] ;then J=$(getconf _NPROCESSORS_ONLN); else J=$1; fi

eval `scram runtime -sh`
cd ${LOCALRT}/tmp/
touch check-start
#touch function-checker.txt.unsorted class-checker.txt.unsorted
cd ${LOCALRT}/src/Utilities/StaticAnalyzers
scram b -j $J
cd ${LOCALRT}/
export USER_CXXFLAGS="-DEDM_ML_DEBUG -w"
export USER_LLVM_CHECKERS="-enable-checker threadsafety -enable-checker optional.ClassChecker -enable-checker cms -enable-checker deprecated -disable-checker cms.FunctionDumper"
export BUILD_LOG=yes
scram b -k -j $J checker  SCRAM_IGNORE_PACKAGES=Fireworks/% SCRAM_IGNORE_SUBDIRS=test > ${LOCALRT}/tmp/class+function-checker.log 2>&1
if [ "${JENKINS_HOME}" != "" ]; then
  BUILD_LOG_DIR=${LOCALRT}/tmp/${SCRAM_ARCH}/cache/log
  scram build outputlog >> ${LOCALRT}/tmp/class+function-checker.log 2>&1 || true
  ${CMS_BOT_DIR}/buildLogAnalyzer.py ${ANALOG_OPT} --logDir ${BUILD_LOG_DIR}/src > ${WORKSPACE}/build-logs.log 2>&1  || true
  if [ -d ${BUILD_LOG_DIR}/html ] ; then mv ${BUILD_LOG_DIR}/html ${WORKSPACE}/build-logs ; fi
fi
cd ${LOCALRT}/tmp/
touch check-end
sort -u < class-checker.txt.unsorted | grep -e"^data class">class-checker.txt
sort -u < function-checker.txt.unsorted >function-statics-db.txt
cat constcast-checker.txt.unsorted constcastaway-checker.txt.unsorted mutablemember-checker.txt.unsorted | sort -u >const-checker.txt
rm *.txt.unsorted
