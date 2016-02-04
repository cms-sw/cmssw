import FWCore.ParameterSet.Config as cms



recoMuonFromPFProducer = cms.EDProducer("RecoMuonFromPFProducer",

    particles = cms.InputTag("particleFlow"),

    verbose = cms.untracked.bool(False),
 
)



