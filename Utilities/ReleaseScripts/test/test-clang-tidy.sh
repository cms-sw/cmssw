#!/bin/bash -ex
clang-tidy -export-fixes test-clang-tidy.cc.yaml -header-filter "src/.*" $CMSSW_BASE/src/Utilities/ReleaseScripts/test/test-clang-tidy.cc
sed -i -e "s|$CMSSW_BASE/src/||" test-clang-tidy.cc.yaml
sed -i -e '/^\s\s*BuildDirectory/d;/^\s\s*Level:/d' test-clang-tidy.cc.yaml
diff -u test-clang-tidy.cc.yaml $CMSSW_BASE/src/Utilities/ReleaseScripts/test/test-clang-tidy.cc.yaml
