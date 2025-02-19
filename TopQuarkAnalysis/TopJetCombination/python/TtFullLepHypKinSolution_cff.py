import FWCore.ParameterSet.Config as cms

#
# produce kinSolution hypothesis with all necessary 
# ingredients
#

## std sequence to perform kinematic fit
import TopQuarkAnalysis.TopKinFitter.TtFullLepKinSolutionProducer_cfi
kinSolutionTtFullLepEventHypothesis = TopQuarkAnalysis.TopKinFitter.TtFullLepKinSolutionProducer_cfi.kinSolutionTtFullLepEvent.clone()
kinSolutionTtFullLepEventHypothesis.mumuChannel=True

## configure kinSolution hypothesis
from TopQuarkAnalysis.TopJetCombination.TtFullLepHypKinSolution_cfi import *

## make hypothesis
makeHypothesis_kinSolution = cms.Sequence(kinSolutionTtFullLepEventHypothesis *
                                          ttFullLepHypKinSolution
					 )

