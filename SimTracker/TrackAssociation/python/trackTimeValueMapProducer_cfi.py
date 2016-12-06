import FWCore.ParameterSet.Config as cms

trackTimeValueMapProducer = cms.EDProducer(
    'TrackTimeValueMapProducer',
    trackSrc = cms.InputTag('generalTracks'),
    gsfTrackSrc = cms.InputTag('electronGsfTracks'),
    trackingParticleSrc = cms.InputTag('mix:MergedTrackTruth'),
    trackingVertexSrc = cms.InputTag('mix:MergedTrackTruth'),
    associators = cms.VInputTag(cms.InputTag('quickTrackAssociatorByHits')),
    resolutionModels = cms.VPSet( cms.PSet( modelName = cms.string('ConfigurableFlatResolutionModel'),
                                            resolutionInNs = cms.double(0.030) ),
                                  cms.PSet( modelName = cms.string('PerfectResolutionModel') ) )
    )
