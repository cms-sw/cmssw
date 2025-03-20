import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator

hltTiclIterLabels = ["hltTiclTrackstersCLUE3DHigh", "hltTiclTrackstersMerge"]

hltAssociatorInstances = []

for labelts in hltTiclIterLabels:
    for labelsts in ['hltTiclSimTracksters', 'hltTiclSimTrackstersfromCPs']:
        hltAssociatorInstances.append(labelts+'To'+labelsts)
        hltAssociatorInstances.append(labelsts+'To'+labelts)

hltHgcalValidator = _hgcalValidator.clone(
    LayerClustersInputMask = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclSimTracksters:fromCPs", "hltTiclSimTracksters"),
    label_tst = cms.VInputTag(*[cms.InputTag(label) for label in hltTiclIterLabels] + [cms.InputTag("hltTiclSimTracksters", "fromCPs"), cms.InputTag("hltTiclSimTracksters")]),
    allTracksterTracksterAssociatorsLabels = cms.VInputTag( *[cms.InputTag('hltAllTrackstersToSimTrackstersAssociationsByLCs:'+associator) for associator in hltAssociatorInstances] ),
    allTracksterTracksterByHitsAssociatorsLabels = cms.VInputTag( *[cms.InputTag('allTrackstersToSimTrackstersAssociationsByHits:'+associator) for associator in hltAssociatorInstances] ),
    associator = cms.untracked.InputTag("hltLayerClusterCaloParticleAssociationProducer"),
    associatorSim = cms.untracked.InputTag("hltLayerClusterSimClusterAssociationProducer"),
    dirName = cms.string('HLT/HGCAL/HGCalValidator/'),
    hits = cms.VInputTag("hltHGCalRecHit:HGCEERecHits", "hltHGCalRecHit:HGCHEFRecHits", "hltHGCalRecHit:HGCHEBRecHits"),
    hitMap = cms.InputTag("hltRecHitMapProducer","hgcalRecHitMap"),
    simTrackstersMap = cms.InputTag("hltTiclSimTracksters"),
    label_layerClusterPlots = cms.InputTag("hltHgcalMergeLayerClusters"),
    label_lcl = cms.InputTag("hltHgcalMergeLayerClusters"),
    label_simTS = cms.InputTag("hltTiclSimTracksters"),
    label_simTSFromCP = cms.InputTag("hltTiclSimTracksters","fromCPs"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    simClustersToCaloParticlesMap = cms.InputTag("SimClusterToCaloParticleAssociation","simClusterToCaloParticleMap"),
    simTiclCandidates = cms.InputTag("hltTiclSimTracksters"),
    ticlCandidates = cms.string('hltTiclCandidate'),
    ticlTrackstersMerge = cms.InputTag("hltTiclTrackstersMerge"),
)
