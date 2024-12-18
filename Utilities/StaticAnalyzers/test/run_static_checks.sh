#!/bin/bash -xe
[ -z $WORKSPACE ] && (echo "WORKSPACE not set!"; exit 1)
[ -z CMSSW_BASE ] && (echo "CMSSW_BASE not set!"; exit 1)
: ${NCPU2:=1}

export CMS_BOT_DIR=$WORKSPACE/cms-bot

rm -rf $WORKSPACE/llvm-analysis-test; mkdir $WORKSPACE/llvm-analysis-test
pushd $CMSSW_BASE/src/Utilities/StaticAnalyzers/test
USER_CXXFLAGS='-Wno-register -DEDM_ML_DEBUG -w' SCRAM_IGNORE_PACKAGES="Fireworks/%" USER_LLVM_CHECKERS="-enable-checker threadsafety -enable-checker cms -enable-checker deprecated -disable-checker cms.FunctionDumper" \
    scram b -v -k -j ${NCPU2} checker >$WORKSPACE/llvm-analysis-test/runStaticChecks.log 2>&1 || true
popd
grep $WORKSPACE/llvm-analysis-test/runStaticChecks.log -e 'warning:' | grep -v "gmake" | sort > $WORKSPACE/llvm-analysis-test/probe.log
diff -u $CMSSW_BASE/src/Utilities/StaticAnalyzers/test/reference.log $WORKSPACE/llvm-analysis-test/probe.log
