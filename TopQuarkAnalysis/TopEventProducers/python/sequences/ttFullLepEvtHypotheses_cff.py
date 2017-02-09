import FWCore.ParameterSet.Config as cms

#
# produce ttFullLep event hypotheses
#

## genMatch hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullLepHypGenMatch_cff import *

## kinematic solution hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullLepHypKinSolution_cff import *

## make all considered event hypotheses
makeTtFullLepHypotheses  = cms.Sequence(makeHypothesis_genMatch *
                                        makeHypothesis_kinSolution
                                       )

