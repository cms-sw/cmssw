import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLep event hypotheses
#

## geom hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGeom_cff import *

## wMassDeltaTopMass hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassDeltaTopMass_cff import *

## wMassMaxSumPt hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypWMassMaxSumPt_cff import *

## maxSumPtWMass hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMaxSumPtWMass_cff import *

## genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypGenMatch_cff import *

## mvaDisc hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMVADisc_cff import *

## kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypKinFit_cff import *

## hitFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypHitFit_cff import *

## make all considered event hypotheses
makeTtSemiLepHypothesesTask = cms.Task(
    makeHypothesis_geomTask,
    makeHypothesis_wMassDeltaTopMassTask,
    makeHypothesis_wMassMaxSumPtTask,
    makeHypothesis_maxSumPtWMassTask,
    makeHypothesis_genMatchTask,
    makeHypothesis_mvaDiscTask,
    makeHypothesis_kinFitTask,
    makeHypothesis_hitFitTask
)
makeTtSemiLepHypotheses  = cms.Sequence(makeTtSemiLepHypothesesTask)
