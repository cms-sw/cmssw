#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function testJSON {
    # The JSON document is on one line, using an arbitrary key here to
    # extract all of the documents into a separte file
    grep cmssw_version $1 > test.json
    python3 <<EOF
import json
with open('test.json') as f:
    for line in f:
        d = json.loads(line)
        for k in d.keys():
            if k in ['producer', 'type', 'type_prefix', 'timestamp', 'host']:
                raise RuntimeError("Found restricted keyword %s"%k)
EOF
    RET=$?
    if [ "x$RET" != "x0" ]; then
        echo "Invalid JSON in $1:"
        cat test.json
        exit $RET
    fi
}

LOCAL_TEST_DIR=${CMSSW_BASE}/src/Utilities/StorageFactory/test
LOCAL_TMP_DIR=${CMSSW_BASE}/tmp/${SCRAM_ARCH}

pushd ${LOCAL_TMP_DIR}

#setup files used in tests
cmsRun ${LOCAL_TEST_DIR}/make_test_files_cfg.py &> make_test_files.log || die "cmsRun make_test_files_cfg.py" $?
rm make_test_files.log
cmsRun ${LOCAL_TEST_DIR}/make_2nd_file_cfg.py &> make_2nd_file.log || die "cmsRun make_2nd_file_cfg.py" $?
rm make_2nd_file.log

cmsRun ${LOCAL_TEST_DIR}/test_single_file_statistics_sender_cfg.py &> test_single_file_statistics_sender.log || die "cmsRun test_single_file_statistics_sender_cfg.py" $?
grep -q '"file_lfn":"file:stat_sender_first.root"' test_single_file_statistics_sender.log || die "no StatisticsSenderService output for single file" 1
grep -q "\"cmssw_version\":\"$CMSSW_VERSION\"" test_single_file_statistics_sender.log || die "no StatisticsSenderService output for CMSSW version" 1
testJSON test_single_file_statistics_sender.log
rm test_single_file_statistics_sender.log

cmsRun ${LOCAL_TEST_DIR}/test_multiple_files_file_statistics_sender_cfg.py &> test_multiple_files_file_statistics_sender.log || die "cmsRun test_multiple_files_file_statistics_sender_cfg.py" $?
grep -q '"file_lfn":"file:stat_sender_b.root"' test_multiple_files_file_statistics_sender.log || die "no StatisticsSenderService output for file b in multiple files" 1
grep -q '"file_lfn":"file:stat_sender_c.root"' test_multiple_files_file_statistics_sender.log || die "no StatisticsSenderService output for file c in multiple files" 1
grep -q '"file_lfn":"file:stat_sender_d.root"' test_multiple_files_file_statistics_sender.log || die "no StatisticsSenderService output for file d in multiple files" 1
grep -q '"file_lfn":"file:stat_sender_e.root"' test_multiple_files_file_statistics_sender.log || die "no StatisticsSenderService output for file e in multiple files" 1
testJSON test_multiple_files_file_statistics_sender.log
rm test_multiple_files_file_statistics_sender.log

cmsRun ${LOCAL_TEST_DIR}/test_multi_file_statistics_sender_cfg.py &> test_multi_file_statistics_sender.log || die "cmsRun test_multi_file_statistics_sender_cfg.py" $?
grep -q '"file_lfn":"file:stat_sender_first.root"' test_multi_file_statistics_sender.log || die "no StatisticsSenderService output for file first in multi file" 1
grep -q '"file_lfn":"file:stat_sender_second.root"' test_multi_file_statistics_sender.log || die "no StatisticsSenderService output for file second in multi file" 1
testJSON test_multi_file_statistics_sender.log
rm test_multi_file_statistics_sender.log

cmsRun ${LOCAL_TEST_DIR}/test_secondary_file_statistics_sender_cfg.py &> test_secondary_file_statistics_sender.log || die "cmsRun test_secondary_file_statistics_sender_cfg.py" $?
grep -q '"file_lfn":"file:stat_sender_first.root"' test_secondary_file_statistics_sender.log || die "no StatisticsSenderService output for file 'first' in secondary files"  1
grep -q '"file_lfn":"file:stat_sender_b.root"' test_secondary_file_statistics_sender.log || die "no StatisticsSenderService output for file 'b' in secondary files"  1
grep -q '"file_lfn":"file:stat_sender_c.root"' test_secondary_file_statistics_sender.log || die "no StatisticsSenderService output for file 'c' in secondary files"  1
grep -q '"file_lfn":"file:stat_sender_d.root"' test_secondary_file_statistics_sender.log || die "no StatisticsSenderService output for file 'd' in secondary files"  1
grep -q '"file_lfn":"file:stat_sender_e.root"' test_secondary_file_statistics_sender.log || die "no StatisticsSenderService output for file 'e' in secondary files"  1
testJSON test_secondary_file_statistics_sender.log
rm test_secondary_file_statistics_sender.log

popd
