import FWCore.ParameterSet.Config as cms

from SimRomanPot.CTPPSOpticsParameterisation.ctppsDetectorPackages_cff import detectorPackages_2016PreTS2
from SimRomanPot.CTPPSOpticsParameterisation.lhcBeamConditions_cff import lhcBeamConditions_2016PreTS2

plotsProducer = cms.EDAnalyzer('PlotsProducer',
    genProtonsTag = cms.InputTag('lhcBeamProducer', 'unsmeared'),
    recoProtons45Tag = cms.InputTag('ctppsOpticsReconstruction', 'sector45'),
    recoProtons56Tag = cms.InputTag('ctppsOpticsReconstruction', 'sector56'),
    potsHitsTag = cms.InputTag('ctppsOpticsParameterisation'),
    detectorPackages = detectorPackages_2016PreTS2,
    beamConditions = lhcBeamConditions_2016PreTS2,
)
