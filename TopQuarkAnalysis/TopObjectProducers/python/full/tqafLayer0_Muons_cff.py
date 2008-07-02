import FWCore.ParameterSet.Config as cms

allLayer0Muons.muonSource = 'muons'
muonMatch.src = 'allLayer0Muons'
muonMatch.matched = 'genParticles'
muonMatch.maxDeltaR = 0.5
muonMatch.maxDPtRel = 0.5
muonMatch.resolveAmbiguities = True
muonMatch.resolveByMatchQuality = False
muonMatch.checkCharge = True
muonMatch.mcPdgId = [13]
muonMatch.mcStatus = [1]
allLayer0Muons.isolation.tracker.src = 'patAODMuonIsolations:muIsoDepositTk'
allLayer0Muons.isolation.tracker.deltaR = 0.3
allLayer0Muons.isolation.tracker.cut = 2.0
allLayer0Muons.isolation.hcal.src = 'patAODMuonIsolations:muIsoDepositCalByAssociatorTowersecal'
allLayer0Muons.isolation.hcal.deltaR = 0.3
allLayer0Muons.isolation.hcal.cut = 2.0
allLayer0Muons.isolation.ecal.src = 'patAODMuonIsolations:muIsoDepositCalByAssociatorTowershcal'
allLayer0Muons.isolation.ecal.deltaR = 0.3
allLayer0Muons.isolation.ecal.cut = 2.0

