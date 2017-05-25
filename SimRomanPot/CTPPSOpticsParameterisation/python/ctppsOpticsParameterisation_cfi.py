import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2
from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

ctppsOpticsParameterisation = cms.EDProducer('CTPPSOpticsParameterisation',
    beamParticlesTag = cms.InputTag('lhcBeamProducer'),
    beamConditions = lhcBeamConditions_2016PreTS2,
    detectorPackages = detectorPackages_2016PreTS2,

    opticsFileBeam1 = cms.FileInPath('SimRomanPot/CTPPSOpticsParameterisation/data/parametrization_6500GeV_0p4_185_reco_beam1.root'),
    opticsFileBeam2 = cms.FileInPath('SimRomanPot/CTPPSOpticsParameterisation/data/parametrization_6500GeV_0p4_185_reco_beam2.root'),
)
