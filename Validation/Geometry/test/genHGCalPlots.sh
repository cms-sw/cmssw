#!/bin/bash -ex

geom=Extended2026D92
VGEO_DIR=${CMSSW_BASE}/src/Validation/Geometry/test
TEST_DIR=.

cmsRun ${VGEO_DIR}/single_neutrino_cfg.py nEvents=1 >$TEST_DIR/single_neutrino_cfg.log 2>&1

python_cmd="python2"
python3 -c "from FWCore.PythonFramework.CmsRun import CmsRun" 2>/dev/null && python_cmd="python3"
${python_cmd} ${VGEO_DIR}/runP_HGCal_cfg.py geom=${geom} label=HGCal >$TEST_DIR/runP_HGCal_cfg_${geom}.log 2>&1
python3 ${VGEO_DIR}/MaterialBudgetHGCal.py -s -d HGCal
