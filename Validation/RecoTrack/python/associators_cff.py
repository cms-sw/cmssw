import FWCore.ParameterSet.Config as cms

#### TrackAssociation
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
import SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi

hltTPClusterProducer = cms.EDProducer("ClusterTPAssociationProducer",
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    verbose = cms.bool(False),
    pixelClusterSrc = cms.InputTag("hltSiPixelClusters"),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    trackingParticleSrc = cms.InputTag("mix","MergedTrackTruth"),
#    stripClusterSrc = cms.InputTag("hltSiStripClusters"),
    stripClusterSrc = cms.InputTag("hltSiStripRawToClustersFacility"),                                      
    simTrackSrc = cms.InputTag("g4SimHits")
)

hltTrackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
hltTrackAssociatorByHits.cluster2TPSrc            = cms.InputTag("hltTPClusterProducer")
hltTrackAssociatorByHits.UseGrouped               = cms.bool( False )
hltTrackAssociatorByHits.UseSplitting             = cms.bool( False )
hltTrackAssociatorByHits.ThreeHitTracksAreSpecial = cms.bool( False )

hltTrackAssociatorByDeltaR = SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi.trackAssociatorByPosition.clone()
hltTrackAssociatorByDeltaR.method             = cms.string('momdr')
hltTrackAssociatorByDeltaR.QCut               = cms.double(0.5)
hltTrackAssociatorByDeltaR.ConsiderAllSimHits = cms.bool(True)


tpToHLTpixelTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltPixelTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter0tracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter0PFlowCtfWithMaterialTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
#    associator = cms.InputTag('hltTrackAssociatorByDeltaR'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter0HPtracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter0PFlowTrackSelectionHighPurity"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
#    associator = cms.InputTag('hltTrackAssociatorByDeltaR'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter1tracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter1PFlowCtfWithMaterialTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter1HPtracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter1PFlowTrackSelectionHighPurity"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter1MergedTracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter1Merged"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter2tracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter2PFlowCtfWithMaterialTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter2HPtracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter2PFlowTrackSelectionHighPurity"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter2MergedTracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter2Merged"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter3tracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter3PFlowCtfWithMaterialTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter3HPtracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter3PFlowTrackSelectionHighPurity"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter3MergedTracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter3Merged"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter4tracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter4PFlowCtfWithMaterialTracks"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter4HPtracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter4PFlowTrackSelectionHighPurity"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter4MergedTracksAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter4Merged"),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTtracksAssociationSequence = cms.Sequence(
    hltTrackAssociatorByHits +
    tpToHLTpixelTrackAssociation +
#    tpToHLTiter0tracksAssociation +
    tpToHLTiter0HPtracksAssociation +
#    tpToHLTiter1tracksAssociation +
    tpToHLTiter1HPtracksAssociation +
    tpToHLTiter1MergedTracksAssociation +
#    tpToHLTiter2tracksAssociation +
    tpToHLTiter2HPtracksAssociation +
    tpToHLTiter2MergedTracksAssociation +
#    tpToHLTiter3tracksAssociation +
    tpToHLTiter3HPtracksAssociation +
    tpToHLTiter3MergedTracksAssociation +
#    tpToHLTiter4tracksAssociation +
    tpToHLTiter4HPtracksAssociation +
    tpToHLTiter4MergedTracksAssociation
)
