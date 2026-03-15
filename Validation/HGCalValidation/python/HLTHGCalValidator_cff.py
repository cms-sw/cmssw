import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels as _hltTiclIterLabels

hltAssociatorInstances = []
for labelts in _hltTiclIterLabels:
    for labelsts in ['hltTiclSimTracksters', 'hltTiclSimTrackstersfromCPs']:
        hltAssociatorInstances.append(labelts+'To'+labelsts)
        hltAssociatorInstances.append(labelsts+'To'+labelts)

# Default is TICLv5
hltHgcalValidator = _hgcalValidator.clone(
    LayerClustersInputMask = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclSimTracksters:fromCPs", "hltTiclSimTracksters"),
    label_tst = cms.VInputTag(*[cms.InputTag(label) for label in _hltTiclIterLabels] + [cms.InputTag("hltTiclSimTracksters", "fromCPs"), cms.InputTag("hltTiclSimTracksters")]),
    allTracksterTracksterAssociatorsLabels = cms.VInputTag( *[cms.InputTag('hltAllTrackstersToSimTrackstersAssociationsByLCs:'+associator) for associator in hltAssociatorInstances] ),
    allTracksterTracksterByHitsAssociatorsLabels = cms.VInputTag( *[cms.InputTag('hltAllTrackstersToSimTrackstersAssociationsByHits:'+associator) for associator in hltAssociatorInstances] ),
    associator = cms.untracked.InputTag("hltLayerClusterCaloParticleAssociationProducer"),
    associatorSim = cms.untracked.InputTag("hltLayerClusterSimClusterAssociationProducer"),
    dirName = cms.string('HLT/HGCAL/HGCalValidator/'),
    hits = cms.InputTag("hltRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    hitMap = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
    simTrackstersMap = cms.InputTag("hltTiclSimTracksters"),
    label_layerClustersPlots = cms.string("hltHgcalMergeLayerClusters"),
    label_lcl = cms.InputTag("hltMergeLayerClusters"),
    label_simTS = cms.InputTag("hltTiclSimTracksters"),
    label_simTSFromCP = cms.InputTag("hltTiclSimTracksters","fromCPs"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    simClustersToCaloParticlesMap = cms.InputTag("SimClusterToCaloParticleAssociation","simClusterToCaloParticleMap"),
    simTiclCandidates = cms.InputTag("hltTiclSimTracksters"),
    ticlCandidates = cms.string('hltTiclCandidate'),
    ticlTrackstersMerge = cms.InputTag("hltTiclCandidate"),
    mergeRecoToSimAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs","hltTiclCandidateTohltTiclSimTrackstersfromCPs"),
    mergeSimToRecoAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs","hltTiclSimTrackstersfromCPsTohltTiclCandidate"),
)
