import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabelsPSet as _hltTiclIterLabelsPSet

hltAssociatorInstances = []
for labelts in _hltTiclIterLabelsPSet.labels:
    for labelsts in ["hltTiclSimTrackstersfromBoundarySimCluster", 'hltTiclSimTrackstersfromCaloParticle']:
        hltAssociatorInstances.append(labelts+'To'+labelsts)
        hltAssociatorInstances.append(labelsts+'To'+labelts)

# Default is TICLv5
hltHgcalValidator = _hgcalValidator.clone(
    LayerClustersInputMask = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclSimTracksters:fromCaloParticle", "hltTiclSimTracksters:fromBoundarySimCluster"), # "hltTiclSimTracksters:fromLegacySimCluster", 
    label_tst = cms.VInputTag(*[cms.InputTag(label) for label in _hltTiclIterLabelsPSet.labels] + [cms.InputTag("hltTiclSimTracksters", "fromCaloParticle"), cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster")]), # cms.InputTag("hltTiclSimTracksters", "fromLegacySimCluster"),
    allTracksterTracksterAssociatorsLabels = cms.VInputTag( *[cms.InputTag('hltAllTrackstersToSimTrackstersAssociationsByLCs:'+associator) for associator in hltAssociatorInstances] ),
    allTracksterTracksterByHitsAssociatorsLabels = cms.VInputTag( *[cms.InputTag('hltAllTrackstersToSimTrackstersAssociationsByHits:'+associator) for associator in hltAssociatorInstances] ),
    associator = cms.untracked.InputTag("hltHGCalLayerClusterCaloParticleAssociation"),
    associatorSim = cms.untracked.InputTag("hltHGCalLayerClusterSimClusterAssociation"),
    dirName = cms.string('HLT/HGCAL/HGCalValidator/'),
    hits = cms.InputTag("hltHGCalRecHitMapProducer", "RefProdVectorHGCRecHitCollection"),
    hitMap = cms.InputTag("hltHGCalRecHitMapProducer","hgcalRecHitMap"),
    simTrackstersMap = cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
    label_layerClustersPlots = cms.string("hltHgcalMergeLayerClusters"),
    label_lcl = cms.InputTag("hltMergeLayerClusters"),
    label_simTS = cms.InputTag("hltTiclSimTracksters", "fromBoundarySimCluster"),
    label_simTSFromCP = cms.InputTag("hltTiclSimTracksters","fromCaloParticle"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    simTiclCandidates = cms.InputTag("hltTiclSimTICLCandidatesFromBoundary"),
    ticlCandidates = cms.string('hltTiclCandidate'),
    ticlTrackstersMerge = cms.InputTag("hltTiclCandidate"),
    mergeRecoToSimAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs","hltTiclCandidateTohltTiclSimTrackstersfromCaloParticle"),
    mergeSimToRecoAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs","hltTiclSimTrackstersfromCaloParticleTohltTiclCandidate"),
)

