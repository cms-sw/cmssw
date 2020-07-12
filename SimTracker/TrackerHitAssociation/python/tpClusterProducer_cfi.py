import FWCore.ParameterSet.Config as cms

from SimTracker.TrackerHitAssociation.tpClusterProducerDefault_cfi import tpClusterProducerDefault as _tpClusterProducerDefault

tpClusterProducer = _tpClusterProducerDefault.clone()

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify( 
    tpClusterProducer,
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel"),
    phase2OTSimLinkSrc = cms.InputTag("simSiPixelDigis","Tracker")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(tpClusterProducer,
    trackingParticleSrc = "mixData:MergedTrackTruth",
    pixelSimLinkSrc = "mixData:PixelDigiSimLink",
    stripSimLinkSrc = "mixData:StripDigiSimLink",
    phase2OTSimLinkSrc = "mixData:Phase2OTDigiSimLink",
)

from SimTracker.TrackerHitAssociation.tpClusterProducerCUDADefault_cfi import tpClusterProducerCUDADefault as _tpClusterProducerCUDA
tpClusterProducerCUDA = _tpClusterProducerCUDA.clone()
