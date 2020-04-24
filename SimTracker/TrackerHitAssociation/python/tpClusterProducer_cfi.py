import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerHitAssociation.tpClusterProducerDefault_cfi import *

tpClusterProducer = tpClusterProducerDefault.clone()

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( 
    tpClusterProducer,
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker")
)
