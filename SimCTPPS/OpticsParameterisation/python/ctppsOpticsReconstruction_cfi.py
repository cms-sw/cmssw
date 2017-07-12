import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2
from SimCTPPS.OpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

ctppsOpticsReconstruction = cms.EDProducer('CTPPSOpticsReconstruction',
    potsHitsTag = cms.InputTag('ctppsOpticsParameterisation'),

    beamConditions = lhcBeamConditions_2016PreTS2,
    detectorPackages = detectorPackages_2016PreTS2,

    checkApertures = cms.bool(True),
    invertBeamCoordinatesSystem = cms.bool(True),

    opticsFileBeam1 = cms.FileInPath('SimCTPPS/OpticsParameterisation/data/parametrization_6500GeV_0p4_185_reco_beam1.root'),
    opticsFileBeam2 = cms.FileInPath('SimCTPPS/OpticsParameterisation/data/parametrization_6500GeV_0p4_185_reco_beam2.root'),
)
