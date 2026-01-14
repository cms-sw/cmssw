#!/bin/bash

# possible exit codes
SUCCESS=0
FAILURE="fail"
DATAFORMATS_CHANGED=40
POLICY_VIOLATION=41

# added test comments
# added
# line added
function die { echo Failure $1: status $2 ; exit $2 ; }
function checkExitCode {
    if [ "$1" == "$FAILURE" ]; then
        if [[ "$2" = @("$SUCCESS"|"$DATAFORMATS_CHANGES"|"$POLICY_VIOLATIONS") ]]; then
            echo "checkDictionaryUpdate.py $3: expected failure exit code, got $2"
            exit 1
        fi
    elif [ "$1" != "$2" ]; then
        echo "checkDictionaryUpdate.py $3: expected exit code $1, got $2"
        exit 1
    fi
}

JSONPATH=${SCRAM_TEST_PATH}/checkDictionaryUpdate

checkDictionaryUpdate.py --baseline ${JSONPATH}/dumpClassVersion_baseline.json || die "checkDictionaryUpdate.py baseline" $?
checkDictionaryUpdate.py --baseline ${JSONPATH}/dumpClassVersion_baseline.json --pr ${JSONPATH}/dumpClassVersion_baseline.json || die "checkDictionaryUpdate.py baseline baseline" $?

checkDictionaryUpdate.py
RET=$?
checkExitCode ${FAILURE} $RET ""

checkDictionaryUpdate.py --baseline ${JSONPATH}/dumpClassVersion_baseline.json --pr ${JSONPATH}/dumpClassVersion_versionUpdate.json
RET=$?
checkExitCode ${DATAFORMATS_CHANGED} $RET "baseline versionUpdate"

checkDictionaryUpdate.py --baseline ${JSONPATH}/dumpClassVersion_baseline.json --pr ${JSONPATH}/dumpClassVersion_newClass.json
RET=$?
checkExitCode ${DATAFORMATS_CHANGED} $RET "baseline newClass"

checkDictionaryUpdate.py --baseline ${JSONPATH}/dumpClassVersion_baseline.json --pr ${JSONPATH}/dumpClassVersion_removeClass.json
RET=$?
checkExitCode ${DATAFORMATS_CHANGED} $RET "baseline removeClass"
