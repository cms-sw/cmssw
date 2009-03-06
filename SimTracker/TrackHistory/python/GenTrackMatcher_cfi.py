import FWCore.ParameterSet.Config as cms

generalGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    trackingParticleModule = cms.string('trackingtruthprod'),
    recoTrackModule = cms.string('ctfWithMaterialTracks'),
    associationModule = cms.string('TrackAssociatorByHits'),
    trackingParticleProduct = cms.string(''),
    genParticles = cms.string('genParticles'),
    bestMatchByMaxValue = cms.bool(True)
)

globalMuonsGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    trackingParticleModule = cms.string('trackingtruthprod'),
    recoTrackModule = cms.string('globalMuons'),
    associationModule = cms.string('TrackAssociatorByHits'),
    trackingParticleProduct = cms.string(''),
    genParticles = cms.string('genParticles'),
    bestMatchByMaxValue = cms.bool(True)
)

standAloneMuonsGenTrackMatcher = cms.EDFilter("GenTrackMatcher",
    trackingParticleModule = cms.string('trackingtruthprod'),
    recoTrackModule = cms.string('standAloneMuons'),
    associationModule = cms.string('TrackAssociatorByHits'),
    trackingParticleProduct = cms.string(''),
    genParticles = cms.string('genParticles'),
    bestMatchByMaxValue = cms.bool(True)
)

genTrackMatcher = cms.Sequence(generalGenTrackMatcher*globalMuonsGenTrackMatcher*standAloneMuonsGenTrackMatcher)

