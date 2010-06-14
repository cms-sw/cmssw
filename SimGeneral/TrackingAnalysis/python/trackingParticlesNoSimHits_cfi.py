import FWCore.ParameterSet.Config as cms

import SimGeneral.TrackingAnalysis.trackingParticles_cfi 
mergedtruthNoSimHits = SimGeneral.TrackingAnalysis.trackingParticles_cfi.mergedtruth.clone(
    simHitCollections = cms.PSet(
        pixel = cms.vstring(),
        tracker = cms.vstring(),
        muon = cms.vstring(),
    )
)

trackingParticlesNoSimHits = cms.Sequence(mergedtruthNoSimHits)

