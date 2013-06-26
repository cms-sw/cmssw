import FWCore.ParameterSet.Config as cms

#
# produce hitFit hypothesis with all necessary 
# ingredients
#

## std sequence to perform kinematic fit
import TopQuarkAnalysis.TopHitFit.TtSemiLepHitFitProducer_Muons_cfi
hitFitTtSemiLepEventHypothesis = TopQuarkAnalysis.TopHitFit.TtSemiLepHitFitProducer_Muons_cfi.hitFitTtSemiLepEvent.clone()

## configure hitFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypHitFit_cfi import *

## make hypothesis
makeHypothesis_hitFit = cms.Sequence(hitFitTtSemiLepEventHypothesis *
                                     ttSemiLepHypHitFit)

