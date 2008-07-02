import FWCore.ParameterSet.Config as cms

allLayer1Taus.tauSource = 'allLayer0Taus'
allLayer1Taus.addGenMatch = True
allLayer1Taus.genParticleMatch = 'tauMatch'
allLayer1Taus.addTrigMatch = True
allLayer1Taus.trigPrimMatch = ['tauTrigMatchHLT1Tau']
allLayer1Taus.addResolutions = True
allLayer1Taus.useNNResolutions = True
allLayer1Taus.tauResoFile = 'PhysicsTools/PatUtils/data/Resolutions_tau.root'
selectedLayer1Taus.src = 'allLayer1Taus'
selectedLayer1Taus.cut = 'pt > 15. & abs(eta) < 2.4'

