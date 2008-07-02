import FWCore.ParameterSet.Config as cms

allLayer0Taus.tauSource = 'pfRecoTauProducer'
allLayer0Taus.tauDiscriminatorSource = 'pfRecoTauDiscriminationByIsolation'
tauMatch.src = 'allLayer0Taus'
tauMatch.matched = 'genParticles'
tauMatch.maxDeltaR = 5
tauMatch.maxDPtRel = 99
tauMatch.resolveAmbiguities = True
tauMatch.resolveByMatchQuality = False
tauMatch.checkCharge = True
tauMatch.mcPdgId = [15]
tauMatch.mcStatus = [2]

