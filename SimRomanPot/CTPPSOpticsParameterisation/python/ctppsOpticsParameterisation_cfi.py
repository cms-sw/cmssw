import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2

ctppsOpticsParameterisation = cms.EDProducer('CTPPSOpticsParameterisation',
    beamParticlesTag = cms.InputTag('lhcBeamProducer'),
    detectorPackages = detectorPackages_2016PreTS2,

    opticsFileBeam1 = cms.FileInPath('parametrisations/version4-vale1/beam1/parametrization_6500GeV_0p4_185_reco.root'),
    opticsFileBeam2 = cms.FileInPath('parametrisations/version4-vale1/beam2/parametrization_6500GeV_0p4_185_reco.root'),
)
