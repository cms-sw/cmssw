import FWCore.ParameterSet.Config as cms
import SimTracker.TrackAssociation.digiSimLinkPrunerDefault_cfi as _mod

prunedDigiSimLinks = _mod.digiSimLinkPrunerDefault.clone(
    stripSimLinkSrc = "simSiStripDigis",
    trackingParticles = "prunedTrackingParticles"
)

_prunedDigiSimLinks_phase2 = _mod.digiSimLinkPrunerDefault.clone(
    pixelSimLinkSrc = "simSiPixelDigis:Pixel",
    phase2OTSimLinkSrc = "simSiPixelDigis:Tracker",
    trackingParticles = "prunedTrackingParticles"
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith( 
    prunedDigiSimLinks,
    _prunedDigiSimLinks_phase2
)
