import FWCore.ParameterSet.Config as cms

# PATLayer0 Tau input
allLayer0Taus.tauSource = 'pfRecoTauProducer'
allLayer0Taus.tauDiscriminatorSource = 'pfRecoTauDiscriminationByIsolation'
# PATLayer0 Tau matching
tauMatch.src = 'allLayer0Taus'
tauMatch.matched = 'genParticles'
tauMatch.maxDeltaR = 5
tauMatch.maxDPtRel = 99
tauMatch.resolveAmbiguities = True
tauMatch.resolveByMatchQuality = False
tauMatch.checkCharge = True
tauMatch.mcPdgId = [15]
tauMatch.mcStatus = [2]

