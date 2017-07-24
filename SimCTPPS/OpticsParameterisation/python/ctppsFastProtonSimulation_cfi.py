import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2

ctppsFastProtonSimulation = cms.EDProducer('CTPPSFastProtonSimulation',
    beamParticlesTag = cms.InputTag('generator', 'unsmeared'),
    beamConditions = lhcBeamConditions_2016PreTS2,

    sqrtS = cms.double(13.0e3),

    # crossing angle
    halfCrossingAngleSector45 = cms.double(179.394e-6), # in rad
    halfCrossingAngleSector56 = cms.double(191.541e-6), # in rad

    detectorPackages = detectorPackages_2016PreTS2,
    simulateDetectorsResolution = cms.bool(True),
    produceHitsRelativeToBeam = cms.bool(False),

    roundToPitch = cms.bool(False),
    pitch = cms.double(66.e-3), # mm
    insensitiveMargin = cms.double(34.e-3), # mm

    checkApertures = cms.bool(True),

    opticsFileBeam1 = cms.FileInPath('CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root'),
    opticsFileBeam2 = cms.FileInPath('CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root'),

     # vertex vertical offset in both sectors
    yOffsetSector45 = cms.double(300.0e-6), # in m
    yOffsetSector56 = cms.double(200.0e-6), # in m
)
