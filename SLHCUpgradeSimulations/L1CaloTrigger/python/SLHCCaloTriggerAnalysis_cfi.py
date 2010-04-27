import FWCore.ParameterSet.Config as cms

mcElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep pdgId = 11 & status = 1",
    "keep pdgId = -11 & status =1 ",
    )
)


mcPhotons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep pdgId = 22 & status =1 "
    )
)


mcSequence = cms.Sequence(mcElectrons*
                          mcPhotons
)




SLHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("L1ExtraParticles","EGamma"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)                           

SLHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("L1ExtraParticles","IsoEGamma"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)                           


SLHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("L1ExtraParticles","EGamma"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)                           

SLHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("L1ExtraParticles","IsoEGamma"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)                           



SLHCjets = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("L1ExtraParticles","Jets"),
                           ref    = cms.InputTag("ak5CaloJets"),
                           deltaR = cms.double(0.5),
                           threshold = cms.double(30.)
)                           





analysisSequence = cms.Sequence(SLHCelectrons*
                                SLHCisoElectrons*
                                SLHCphotons*
                                SLHCisoPhotons*
                                SLHCjets
)
