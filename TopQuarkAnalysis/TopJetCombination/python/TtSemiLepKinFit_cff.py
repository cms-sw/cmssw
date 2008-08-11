import FWCore.ParameterSet.Config as cms

#
# produce kinFit hypothesis with all necessary 
# ingredients
#

## std sequence to perform kinematic fit
from TopQuarkAnalysis.TopKinFitter.TtSemiKinFitProducer_Muons_cfi import *

## configure kinFit hypothesis
from TopQuarkAnalysis.TopJetCombination.TtSemiLepKinFit_cfi import *

## make hypothesis
makeHypothesis_kinFit = cms.Sequence(kinFitTtSemiEvent *
                                     ttSemiLepKinFit)

