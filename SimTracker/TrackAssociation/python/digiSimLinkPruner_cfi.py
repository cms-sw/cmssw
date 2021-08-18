import FWCore.ParameterSet.Config as cms

prunedDigiSimLinks = cms.EDProducer("DigiSimLinkPruner",
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    trackingParticles = cms.InputTag('prunedTrackingParticles')
)

_prunedDigiSimLinks_phase2 = cms.EDProducer("DigiSimLinkPruner",
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker"),
    trackingParticles = cms.InputTag('prunedTrackingParticles')
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith( 
    prunedDigiSimLinks,
    _prunedDigiSimLinks_phase2
)
