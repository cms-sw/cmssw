import FWCore.ParameterSet.Config as cms

#
# L1 input
#

## import module
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import allLayer1METs

## configure for tqaf
allLayer1METs.metSource        = 'allLayer0METs'
allLayer1METs.addGenMET        = True
allLayer1METs.genMETSource     = 'genMet'
allLayer1METs.addTrigMatch     = True
allLayer1METs.trigPrimMatch    = ['metTrigMatchHLT1MET65']
allLayer1METs.addResolutions   = True
allLayer1METs.useNNResolutions = False
allLayer1METs.metResoFile      = 'PhysicsTools/PatUtils/data/Resolutions_met.root'

#
# L1 selection
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.metSelector_cfi import selectedLayer1METs

## configure for tqaf
selectedLayer1METs.src         = 'allLayer1METs'
selectedLayer1METs.cut         = 'et > 0.'

