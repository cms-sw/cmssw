import FWCore.ParameterSet.Config as cms

HGVHistoProducerAlgoBlock = cms.PSet(

    minEta = cms.double(-4.5),
    maxEta = cms.double(4.5),
    nintEta = cms.int32(100),
    useFabsEta = cms.bool(False),

)

