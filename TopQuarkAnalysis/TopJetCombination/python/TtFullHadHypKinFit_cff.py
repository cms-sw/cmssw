import FWCore.ParameterSet.Config as cms

#
# produce kinFit hypothesis with all necessary 
# ingredients
#

## std sequence to perform kinematic fit
import TopQuarkAnalysis.TopKinFitter.TtFullHadKinFitProducer_cfi
kinFitTtFullHadEventHypothesis = TopQuarkAnalysis.TopKinFitter.TtFullHadKinFitProducer_cfi.kinFitTtFullHadEvent.clone()

## configure kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullHadHypKinFit_cfi import *

## make hypothesis
makeHypothesis_kinFit = cms.Sequence(kinFitTtFullHadEventHypothesis *
                                     ttFullHadHypKinFit)

