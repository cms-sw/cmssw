#!/bin/bash -ex
clang-tidy -export-fixes $CMSSW_BASE/test-clang-tidy.cc.yaml -header-filter "$CMSSW_BASE/src/.*" $CMSSW_BASE/src/Utilities/ReleaseScripts/test/test-clang-tidy.cc
sed -i -e "s|$CMSSW_BASE/src/||" $CMSSW_BASE/test-clang-tidy.cc.yaml
sed -i -e '/^\s\s*BuildDirectory/d;/^\s\s*Level:/d' $CMSSW_BASE/test-clang-tidy.cc.yaml
diff -u $CMSSW_BASE/test-clang-tidy.cc.yaml $CMSSW_BASE/src/Utilities/ReleaseScripts/test/test-clang-tidy.cc.yaml
rm -f $CMSSW_BASE/test-clang-tidy.cc.yaml
