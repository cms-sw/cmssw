import FWCore.ParameterSet.Config as cms

#
# produce mvaDisc hypothesis with all necessary 
# ingredients
#

## std sequence to compute mva discriminant
from TopQuarkAnalysis.TopJetCombination.TtSemiJetCombMVAComputer_Muons_cff import *

## configure mvaDisc hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypMVADisc_cfi import *

## make hypothesis
makeHypothesis_mvaDisc = cms.Sequence(findTtSemiJetCombMVA *
                                      ttSemiLepHypMVADisc)

