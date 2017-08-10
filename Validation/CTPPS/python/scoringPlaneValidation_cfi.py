import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2

scoringPlaneValidation = cms.EDAnalyzer('ScoringPlaneValidation',
    spTracksTag = cms.InputTag('ctppsLocalTrackLiteProducer'),
    recoTracksTag = cms.InputTag('ctppsFastProtonSimulation', 'scoringPlane'),
    detectorPackages = detectorPackages_2016PreTS2,
)
