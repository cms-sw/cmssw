#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/tqaf_cfg.py || die 'Failure using tqaf_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttDecaySubset_cfg.py || die 'Failure using ttDecaySubset_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttDecaySelection_cfg.py || die 'Failure using ttDecaySelection_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttFullLepEvtBuilder_cfg.py || die 'Failure using ttFullLepEvtBuilder_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttSemiLepEvtBuilder_cfg.py || die 'Failure using ttSemiLepEvtBuilder_cfg.py' $?
