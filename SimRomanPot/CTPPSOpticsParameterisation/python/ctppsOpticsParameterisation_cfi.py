import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2017PreTS2

ctppsOpticsParameterisation = cms.EDProducer('CTPPSOpticsParameterisation',
    beamConditions = lhcBeamConditions_2017PreTS2,

    simulateVertexX = cms.bool(True),
    simulateVertexY = cms.bool(True),
    simulateScatteringAngleX = cms.bool(True),
    simulateScatteringAngleY = cms.bool(True),
    simulateBeamDivergence = cms.bool(True),
    simulateXi = cms.bool(True),
    simulateDetectorsResolution = cms.bool(False),

    opticsFileBeam1 = cms.FileInPath('parametrisations/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root'),
    opticsFileBeam2 = cms.FileInPath('parametrisations/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root'),

    # list of detector packages to simulate
    detectorsList = cms.VPSet(
        cms.PSet(
            name = cms.string("RP1"), #FIXME
            scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
            resolution = cms.double(12.e-6), # RP resolution, m
            minXi = cms.double(0.03),
            maxXi = cms.double(0.17),
        )
    ),
)
