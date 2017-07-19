import FWCore.ParameterSet.Config as cms

from SimCTPPS.OpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2
from SimCTPPS.OpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

paramValidation = cms.EDAnalyzer('ParamValidation',
    #genProtonsTag = cms.InputTag('generator', 'unsmeared'),
    genProtonsTag = cms.InputTag('ctppsFastProtonSimulation', 'smeared'),
    recoProtonsTag = cms.InputTag('ctppsProtonReconstruction'),
    potsTracksTag = cms.InputTag('ctppsLocalTrackLiteProducer'),
    detectorPackages = detectorPackages_2016PreTS2,
    beamConditions = lhcBeamConditions_2016PreTS2,
)
