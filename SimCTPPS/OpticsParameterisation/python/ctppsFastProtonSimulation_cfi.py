import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2

ctppsFastProtonSimulation = cms.EDProducer('CTPPSFastProtonSimulation',
    beamParticlesTag = cms.InputTag('generatorSmeared'),

    sqrtS = cms.double(13.0e3),
    opticsFileBeam1 = cms.FileInPath('CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root'),
    opticsFileBeam2 = cms.FileInPath('CondFormats/CTPPSOpticsObjects/data/2016_preTS2/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root'),

    checkApertures = cms.bool(True),
    produceHitsRelativeToBeam = cms.bool(False),

    # crossing angle
    halfCrossingAngleSector45 = cms.double(179.394e-6), # in rad
    halfCrossingAngleSector56 = cms.double(191.541e-6), # in rad
     # vertex vertical offset in both sectors
    yOffsetSector45 = cms.double(300.0e-6), # in m
    yOffsetSector56 = cms.double(200.0e-6), # in m

    detectorPackages = detectorPackages_2016PreTS2,

    scoringPlaneParams = cms.PSet(
        simulateDetectorsResolution = cms.bool(False),
    ),
    stripsRecHitsParams = cms.PSet(
        roundToPitch = cms.bool(False),
        pitch = cms.double(66.e-3), # mm
        insensitiveMargin = cms.double(34.e-3), # mm
    ),
)

