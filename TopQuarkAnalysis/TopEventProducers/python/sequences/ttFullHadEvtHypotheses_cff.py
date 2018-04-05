import FWCore.ParameterSet.Config as cms

#
# produce ttFullHadLep event hypotheses
#

## genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullHadHypGenMatch_cff import *

## kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullHadHypKinFit_cff import *

## make all considered event hypotheses
makeTtFullHadHypothesesTask = cms.Task(
    makeHypothesis_genMatchTask,
    makeHypothesis_kinFitTask
)
makeTtFullHadHypotheses = cms.Sequence(makeTtFullHadHypothesesTask)
