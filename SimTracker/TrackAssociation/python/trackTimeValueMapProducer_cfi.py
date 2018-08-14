import FWCore.ParameterSet.Config as cms

trackTimeValueMapProducer = cms.EDProducer(
    'TrackTimeValueMapProducer',
    trackSrc = cms.InputTag('generalTracks'),
    trackingParticleSrc = cms.InputTag('mix:MergedTrackTruth'),
    trackingVertexSrc = cms.InputTag('mix:MergedTrackTruth'),
    pileupSummaryInfo = cms.InputTag('addPileupInfo'),
    associators = cms.VInputTag(cms.InputTag('quickTrackAssociatorByHits')),
    resolutionModels = cms.VPSet( cms.PSet( modelName = cms.string('ConfigurableFlatResolutionModel'),
                                            resolutionInNs = cms.double(0.030) ),
                                  cms.PSet( modelName = cms.string('PerfectResolutionModel') ) ),
    etaMin = cms.double(-1.0),
    etaMax = cms.double(3.0),
    ptMin = cms.double(0.7),
    pMin = cms.double(0.7),
    etaMaxForPtThreshold = cms.double(1.5),
    )
