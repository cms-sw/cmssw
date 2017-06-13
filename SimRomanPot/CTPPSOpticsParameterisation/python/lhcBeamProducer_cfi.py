import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

lhcBeamProducer = cms.EDProducer('FlatRandomXiGunProducer',
    PGunParameters = cms.PSet(
        PartID = cms.vint32(2212),
        BeamConditions = lhcBeamConditions_2016PreTS2,
        SqrtS = lhcBeamConditions_2016PreTS2.sqrtS,
        MinXi = cms.double(0.03),
        MaxXi = cms.double(0.17),
        # switches
        ScatteringAngle = cms.double(25.e-6), # in rad
        SimulateVertexX = cms.bool(True),
        SimulateVertexY = cms.bool(True),
        SimulateScatteringAngleX = cms.bool(True),
        SimulateScatteringAngleY = cms.bool(True),
        SimulateBeamDivergence = cms.bool(True),
    )
)
