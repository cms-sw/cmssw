import FWCore.ParameterSet.Config as cms

cutsTPB = cms.EDFilter("BTrackingParticleSelection",
    src = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trackingParticleModule = cms.string('trackingtruthprod'),
    recoTrackModule = cms.string('generalTracks'),
    associationModule = cms.string('TrackAssociatorByHits'),
    trackingParticleProduct = cms.string(''),
    bestMatchByMaxValue = cms.bool(True)
)


