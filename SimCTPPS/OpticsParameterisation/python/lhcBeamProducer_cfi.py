import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

lhcBeamProducer = cms.EDProducer('FlatRandomXiGunProducer',
    PGunParameters = cms.PSet(
        PartID = cms.vint32(2212),
        SqrtS = lhcBeamConditions_2016PreTS2.sqrtS,
        MinXi = cms.double(0.03),
        MaxXi = cms.double(0.17),
    )
)
