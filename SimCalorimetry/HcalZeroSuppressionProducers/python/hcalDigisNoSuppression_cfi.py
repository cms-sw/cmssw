import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDFilter("HcalRealisticZS",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    mode = cms.int32(0),
    HBlevel = cms.int32(-999),
    HElevel = cms.int32(-999),
    HOlevel = cms.int32(-999),
    HFlevel = cms.int32(-999)
)



