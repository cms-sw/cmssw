#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/tqaf_cfg.py || die 'Failure using tqaf_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/tqaf_woGeneratorInfo_cfg.py || die 'Failure using tqaf_woGeneratorInfo_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttDecaySubset_cfg.py || die 'Failure using ttDecaySubset_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttFullHadEvtBuilder_cfg.py || die 'Failure using ttFullHadEvtBuilder_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttFullLepEvtBuilder_cfg.py || die 'Failure using ttFullLepEvtBuilder_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttSemiLepEvtBuilder_cfg.py || die 'Failure using ttSemiLepEvtBuilder_cfg.py' $?

#cmsRun ${LOCAL_TEST_DIR}/stGenEvent_cfg.py || die 'Failure using stGenEvent_cfg.py' $?
