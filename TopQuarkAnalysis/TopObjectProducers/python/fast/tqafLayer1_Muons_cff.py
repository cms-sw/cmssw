import FWCore.ParameterSet.Config as cms

#
# L1 input
#
allLayer1Muons.muonSource       = 'allLayer0Muons'
allLayer1Muons.addGenMatch      = True
allLayer1Muons.genParticleMatch = 'muonMatch'
allLayer1Muons.addTrigMatch     = False
allLayer1Muons.addResolutions   = True
allLayer1Muons.useNNResolutions = False
allLayer1Muons.muonResoFile = 'PhysicsTools/PatUtils/data/Resolutions_muon.root'
allLayer1Muons.isolation.tracker.src = 'layer0MuonIsolations:muParamGlobalIsoDepositTk'
allLayer1Muons.isolation.ecal.src    = 'layer0MuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowersecal'
allLayer1Muons.isolation.hcal.src    = 'layer0MuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowershcal'
allLayer1Muons.isolation.user        = cms.VPSet(cms.PSet(
    placeholder = cms.bool(True)
))
allLayer1Muons.isoDeposits.tracker = 'layer0MuonIsolations:muParamGlobalIsoDepositTk'
allLayer1Muons.isoDeposits.ecal    = 'layer0MuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowersecal'
allLayer1Muons.isoDeposits.hcal    = 'layer0MuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowershcal'
allLayer1Muons.isoDeposits.user    = ['layer0MuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowersho']
allLayer1Muons.addMuonID        = True

#
# L1 selection
#
selectedLayer1Muons.src = 'allLayer1Muons'
selectedLayer1Muons.cut = 'pt > 10. & abs(eta) < 3.0'

