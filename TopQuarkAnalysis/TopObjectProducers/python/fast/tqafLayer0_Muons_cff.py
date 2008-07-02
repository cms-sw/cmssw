import FWCore.ParameterSet.Config as cms

#
# L0 input
#
allLayer0Muons.muonSource = 'paramMuons:ParamGlobalMuons'

#
# genMatch
#
muonMatch.src       = 'allLayer0Muons'
muonMatch.matched   = 'genParticles'
muonMatch.maxDeltaR = 0.5
muonMatch.maxDPtRel = 0.5
muonMatch.resolveAmbiguities    = True
muonMatch.resolveByMatchQuality = False
muonMatch.checkCharge = True
muonMatch.mcPdgId  = [13]
muonMatch.mcStatus =  [1]

#
# isolation
#
patAODMuonIsolations.collection   = 'paramMuons:ParamGlobalMuons'
patAODMuonIsolations.associations = ['muParamGlobalIsoDepositCalByAssociatorTowers:ecal',
                                     'muParamGlobalIsoDepositCalByAssociatorTowers:hcal',
                                     'muParamGlobalIsoDepositCalByAssociatorTowers:ho',
                                     'muParamGlobalIsoDepositTk',
                                     'muParamGlobalIsoDepositJets']
layer0MuonIsolations.associations = ['muParamGlobalIsoDepositCalByAssociatorTowers:ecal',
                                     'muParamGlobalIsoDepositCalByAssociatorTowers:hcal',
                                     'muParamGlobalIsoDepositCalByAssociatorTowers:ho',
                                     'muParamGlobalIsoDepositTk',
                                     'muParamGlobalIsoDepositJets']
allLayer0Muons.isolation.tracker.src = 'patAODMuonIsolations:muParamGlobalIsoDepositTk'
allLayer0Muons.isolation.ecal.src    = 'patAODMuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowersecal'
allLayer0Muons.isolation.hcal.src    = 'patAODMuonIsolations:muParamGlobalIsoDepositCalByAssociatorTowershcal'
allLayer0Muons.isolation.user        = cms.VPSet(
    cms.PSet(
        src    = cms.InputTag("patAODMuonIsolations",
                              "muParamGlobalIsoDepositCalByAssociatorTowersho"),
        deltaR = cms.double(0.3),
        cut    = cms.double(2.0)
    ), 
    cms.PSet(
        src = cms.InputTag("patAODMuonIsolations","muParamGlobalIsoDepositJets"),
        deltaR = cms.double(0.5),
        cut = cms.double(2.0)
    )
)

