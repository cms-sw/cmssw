#!/bin/bash -e
# Quirk pair gun through g4SimHits; checks that exactly two quirk SimTracks carry the hits
function die { echo $1: status $2 ; exit $2; }
T=${SCRAM_TEST_PATH:-$CMSSW_BASE/src/SimG4Core/CustomPhysics/test}
python3 $T/makeQuirkPairHepMC.py --nEvents 2 --output quirkPairUnitTest.hepmc || die "makeQuirkPairHepMC" $?
cmsRun $T/quirk_pairgun_cfg.py --inputFile file:quirkPairUnitTest.hepmc --outputFile quirkPairUnitTest_SIM.root --maxEvents 2 || die "cmsRun quirk_pairgun_cfg.py" $?
python3 $T/checkQuirkSim.py quirkPairUnitTest_SIM.root | tee checkQuirkSim.log
grep -q "RESULT OK" checkQuirkSim.log || die "checkQuirkSim.py" 1
