#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/tqaf_cfg.py || die 'Failure using tqaf_cfg.py' $?

# FIXME: event content broken in only available data RelVal
# cmsRun ${SCRAM_TEST_PATH}/tqaf_woGeneratorInfo_cfg.py || die 'Failure using tqaf_woGeneratorInfo_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttDecaySubset_cfg.py || die 'Failure using ttDecaySubset_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttFullHadEvtBuilder_cfg.py || die 'Failure using ttFullHadEvtBuilder_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttFullLepEvtBuilder_cfg.py || die 'Failure using ttFullLepEvtBuilder_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttSemiLepEvtBuilder_cfg.py || die 'Failure using ttSemiLepEvtBuilder_cfg.py' $?

#cmsRun ${SCRAM_TEST_PATH}/stGenEvent_cfg.py || die 'Failure using stGenEvent_cfg.py' $?
