import FWCore.ParameterSet.Config as cms

#
# L1 input
#
allLayer1Taus.tauSource        = 'allLayer0Taus'
allLayer1Taus.addGenMatch      = True
allLayer1Taus.genParticleMatch = 'tauMatch'
allLayer1Taus.addTrigMatch     = False
allLayer1Taus.addResolutions   = True
allLayer1Taus.useNNResolutions = True
allLayer1Taus.tauResoFile = 'PhysicsTools/PatUtils/data/Resolutions_tau.root'

#
# L1 selection
#
selectedLayer1Taus.src = 'allLayer1Taus'
selectedLayer1Taus.cut = 'pt > 10. & abs(eta) < 3.0'

