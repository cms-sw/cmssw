import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients
#

## maxSumPtWMass hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiLepMaxSumPtWMass_cff import *

## genMatch hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiLepGenMatch_cff import *

## mvaDisc hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiLepMVADisc_cff import *

## kinFit hypotheses
from TopQuarkAnalysis.TopJetCombination.TtSemiLepKinFit_cff import *

## make all considered event hypotheses
makeTtSemiHyps  = cms.Sequence(makeHypothesis_maxSumPtWMass *
                               makeHypothesis_genMatch      *
                               makeHypothesis_mvaDisc       *
                               makeHypothesis_kinFit
                               )

