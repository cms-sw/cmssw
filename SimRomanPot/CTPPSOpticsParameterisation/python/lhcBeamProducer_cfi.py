import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

lhcBeamProducer = cms.EDProducer('LHCBeamProducer',
    numProtons = cms.uint32(1), # number of protons to generate in each event

    beamConditions = lhcBeamConditions_2016PreTS2,

    simulateVertexX = cms.bool(True),
    simulateVertexY = cms.bool(True),

    simulateScatteringAngleX = cms.bool(True),
    simulateScatteringAngleY = cms.bool(True),
    scatteringAngle = cms.double(25.e-6), # in rad

    simulateBeamDivergence = cms.bool(True),

    simulateXi = cms.bool(True),
    minXi = cms.double(0.03),
    maxXi = cms.double(0.17),
)
