#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_make_file_cfg.py || die "cmsRun test_storageproxy_make_file_cfg.py failed" $?

cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py || die "cmsRun test_storageproxy_test_cfg.py failed" $?

cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py --latencyRead || die "cmsRun test_storageproxy_test_cfg.py --latencyRead failed" $?
cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py --latencyWrite || die "cmsRun test_storageproxy_test_cfg.py --latencyWrite failed" $?

cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py --trace || die "cmsRun test_storageproxy_test_cfg.py --trace failed" $?
grep -q "o .* test.root" trace_0.txt || die "File open entry missing in trace_0.txt" $?
grep -q "r " trace_0.txt || die "No read entries in trace_0.txt" $?
grep -q "o .* output.root" trace_1.txt || die "File open entry missing in trace_1.txt" $?
grep -q "w " trace_1.txt || die "No write entries in trace_0.txt" $?

edmStorageTrace.py --summary trace_0.txt | grep -q "Singular reads" || die "No reads in summary for trace_0.txt" $?
edmStorageTrace.py --summary trace_1.txt | grep -q "Singular writes" || die "No reads in summary for trace_1.txt" $?


cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py --trace --latencyRead || die "cmsRun test_storageproxy_test_cfg.py --trace --latencyRead failed" $?
edmStorageTrace.py --summary trace_0.txt | grep -q "Duration .* ms" || die "Read duration has non-ms units in trace_1.txt" $?

cmsRun ${SCRAM_TEST_PATH}/test_storageproxy_test_cfg.py --trace --latencyWrite || die "cmsRun test_storageproxy_test_cfg.py --trace --latencyWrite failed" $?
edmStorageTrace.py --summary trace_1.txt | grep -q "Duration .* ms" || die "Write duration has non-ms trace_1.txt" $?
