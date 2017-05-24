import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2017PreTS2
from SimRomanPot.CTPPSOpticsParameterisation.ctppsDetectorPackages_cff import ctppsDetectorPackages_2017PreTS2

lhcBeamProducer = cms.EDProducer('LHCBeamProducer',
    beamConditions = lhcBeamConditions_2017PreTS2,

    simulateVertexX = cms.bool(True),
    simulateVertexY = cms.bool(True),
    simulateScatteringAngleX = cms.bool(True),
    simulateScatteringAngleY = cms.bool(True),
    simulateBeamDivergence = cms.bool(True),
    simulateXi = cms.bool(True),
    simulateDetectorsResolution = cms.bool(False),

    detectorPackages = ctppsDetectorPackages_2017PreTS2,
)
