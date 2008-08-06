import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients
#

## maxSumPtWMass hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMaxSumPtWMass_cff import *

## genMatch hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisGenMatch_cff import *

## mvaDisc hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisMVADisc_cff import *

## kinFit hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiHypothesisKinFit_cff import *

## make all considered event hypotheses
makeTtSemiHyps  = cms.Sequence(makeHypothesis_maxSumPtWMass *
                               makeHypothesis_genMatch      *
                               makeHypothesis_mvaDisc       *
                               makeHypothesis_kinFit
                               )

