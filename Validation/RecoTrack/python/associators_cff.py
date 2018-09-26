import FWCore.ParameterSet.Config as cms

#### TrackAssociation
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
import SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import tpClusterProducer as _tpClusterProducer
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation as _trackingParticleRecoTrackAsssociation

hltTPClusterProducer = _tpClusterProducer.clone(
    pixelClusterSrc = "hltSiPixelClusters",
#    stripClusterSrc = "hltSiStripClusters",
    stripClusterSrc = "hltSiStripRawToClustersFacility",
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


# Note: the TrackAssociatorEDProducers defined below, and
# tpToHLTtracksAssociationSequence sequence, are not currently needed
# to run MTV for HLT, as it is configured to produce the
# track-TrackingParticle association on the fly. The configuration
# snippets below are, however, kept for reference.
tpToHLTpixelTrackAssociation = _trackingParticleRecoTrackAsssociation.clone(
    label_tr = cms.InputTag("hltPixelTracks"),
    associator = cms.InputTag('hltTrackAssociatorByHits'),
    ignoremissingtrackcollection = cms.untracked.bool(True)
)

tpToHLTiter0tracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter0PFlowCtfWithMaterialTracks"),
#    associator = cms.InputTag('hltTrackAssociatorByDeltaR'),
)

tpToHLTiter0HPtracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter0PFlowTrackSelectionHighPurity"),
#    associator = cms.InputTag('hltTrackAssociatorByDeltaR'),
)

tpToHLTiter1tracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter1PFlowCtfWithMaterialTracks"),
)

tpToHLTiter1HPtracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter1PFlowTrackSelectionHighPurity"),
)

tpToHLTiter1MergedTracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter1Merged"),
)

tpToHLTiter2tracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter2PFlowCtfWithMaterialTracks"),
)

tpToHLTiter2HPtracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter2PFlowTrackSelectionHighPurity"),
)

tpToHLTiter2MergedTracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter2Merged"),
)

tpToHLTiter3tracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter3PFlowCtfWithMaterialTracks"),
)

tpToHLTiter3HPtracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter3PFlowTrackSelectionHighPurity"),
)

tpToHLTiter3MergedTracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter3Merged"),
)

tpToHLTiter4tracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter4PFlowCtfWithMaterialTracks"),
)

tpToHLTiter4HPtracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter4PFlowTrackSelectionHighPurity"),
)

tpToHLTiter4MergedTracksAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltIter4Merged"),
)

tpToHLTgsfTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = cms.InputTag("hltEgammaGsfTracks"),
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
