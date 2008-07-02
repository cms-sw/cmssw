import FWCore.ParameterSet.Config as cms

# PATLayer1 Tau input
allLayer1Taus.tauSource = 'allLayer0Taus'
allLayer1Taus.addGenMatch = True
allLayer1Taus.genParticleMatch = 'tauMatch'
allLayer1Taus.addTrigMatch = False
# replace allLayer1Taus.trigPrimMatch    = { tauTrigMatchHLT1Tau }
allLayer1Taus.addResolutions = True
allLayer1Taus.useNNResolutions = True
allLayer1Taus.tauResoFile = 'PhysicsTools/PatUtils/data/Resolutions_tau.root'
# PATLayer1 Tau selection
selectedLayer1Taus.src = 'allLayer1Taus'
# replace selectedLayer1Taus.cut  = "pt > 15. & abs(eta) < 2.4 & emEnergyFraction<0.9 & eOverP>0.5"
selectedLayer1Taus.cut = 'pt > 15. & abs(eta) < 2.4'

