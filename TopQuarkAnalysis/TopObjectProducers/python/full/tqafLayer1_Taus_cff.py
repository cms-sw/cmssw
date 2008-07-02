import FWCore.ParameterSet.Config as cms

#
# L1 input
#

## import module
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import allLayer1Taus

## configure for tqaf
allLayer1Taus.tauSource        = 'allLayer0Taus'
allLayer1Taus.addGenMatch      = True
allLayer1Taus.genParticleMatch = 'tauMatch'
allLayer1Taus.addTrigMatch     = True
allLayer1Taus.trigPrimMatch    = ['tauTrigMatchHLT1Tau']
allLayer1Taus.addResolutions   = True
allLayer1Taus.useNNResolutions = True
allLayer1Taus.tauResoFile      = 'PhysicsTools/PatUtils/data/Resolutions_tau.root'

#
# L1 selection
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi import selectedLayer1Taus

## configure for tqaf
selectedLayer1Taus.src         = 'allLayer1Taus'
selectedLayer1Taus.cut         = 'pt > 10. & abs(eta) < 3.0'

