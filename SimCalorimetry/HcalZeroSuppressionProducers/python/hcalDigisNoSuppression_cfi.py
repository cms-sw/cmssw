import FWCore.ParameterSet.Config as cms

simHcalDigis = cms.EDProducer("HcalRealisticZS",
    digiLabel = cms.string("simHcalUnsuppressedDigis"),
    markAndPass = cms.bool(False),
    HBlevel = cms.int32(-999),
    HElevel = cms.int32(-999),
    HOlevel = cms.int32(-999),
    HFlevel = cms.int32(-999),
    HBregion = cms.vint32(0,9),
    HEregion = cms.vint32(0,9),
    HOregion = cms.vint32(0,9),
    HFregion = cms.vint32(0,4),
    useConfigZSvalues = cms.int32(1)
)



