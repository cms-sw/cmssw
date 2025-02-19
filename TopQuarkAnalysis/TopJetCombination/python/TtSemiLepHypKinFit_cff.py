import FWCore.ParameterSet.Config as cms

#
# produce kinFit hypothesis with all necessary 
# ingredients
#

## std sequence to perform kinematic fit
import TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi
kinFitTtSemiLepEventHypothesis = TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi.kinFitTtSemiLepEvent.clone()

## configure kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepHypKinFit_cfi import *

## make hypothesis
makeHypothesis_kinFit = cms.Sequence(kinFitTtSemiLepEventHypothesis *
                                     ttSemiLepHypKinFit)

