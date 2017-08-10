import FWCore.ParameterSet.Config as cms

lhcBeamProducer = cms.EDProducer('FlatRandomXiGunProducer',
    PGunParameters = cms.PSet(
        PartID = cms.vint32(2212),
        SqrtS = cms.double(13.e3),
        MinXi = cms.double(0.03),
        MaxXi = cms.double(0.17),
    )
)
