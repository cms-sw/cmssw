import FWCore.ParameterSet.Config as cms

#
# produce geom hypothesis with all necessary 
# ingredients
#

## configure geom hyothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombGeom_cfi import *
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGeom_cfi import *

## make hypothesis
makeHypothesis_geomTask = cms.Task(
    findTtSemiLepJetCombGeom,
    ttSemiLepHypGeom
)
makeHypothesis_geom = cms.Sequence(makeHypothesis_geomTask)
