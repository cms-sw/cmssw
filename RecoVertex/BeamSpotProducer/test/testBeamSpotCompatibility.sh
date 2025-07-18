#!/bin/sh

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

echo "TESTING BeamSpot compatibility check code ..."


echo "========================================"
echo "testing failing comparison"
echo "----------------------------------------"
echo

check_for_failure cmsRun ${SCRAM_TEST_PATH}/testBeamSpotCompatibility_cfg.py

echo "========================================"
echo "testing successful comparison (with warning)"
echo "----------------------------------------"
echo

check_for_success cmsRun ${SCRAM_TEST_PATH}/testBeamSpotCompatibility_cfg.py errorThreshold=100000

echo "========================================"
echo "testing failed comparison (DB from event)"
echo "----------------------------------------"
echo

check_for_failure cmsRun ${SCRAM_TEST_PATH}/testBeamSpotCompatibility_cfg.py dbFromEvent=True
