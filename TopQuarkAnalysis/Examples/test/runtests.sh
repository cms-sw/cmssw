#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/analyzeTopElectron_cfg.py || die 'Failure using analyzeTopElectron_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/analyzeTopJet_cfg.py || die 'Failure using analyzeTopJet_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/analyzeTopMuon_cfg.py || die 'Failure using analyzeTopMuon_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/analyzeTopTau_cfg.py || die 'Failure using analyzeTopTau_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/analyzeTopGenEvent_cfg.py || die 'Failure using analyzeTopGenEvent_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/analyzeTopHypotheses_cfg.py || die 'Failure using analyzeTopHypotheses_cfg.py' $?
