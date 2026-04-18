import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalValidator_cfi import hgcalValidator as _hgcalValidator
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels as _hltTiclIterLabels

hltAssociatorInstances = []

for labelts in _hltTiclIterLabels:
    for labelsts in ['hltTiclSimTracksters', 'hltTiclSimTrackstersfromCPs']:
        hltAssociatorInstances.append(labelts+'To'+labelsts)
        hltAssociatorInstances.append(labelsts+'To'+labelts)

hltHgcalValidator = _hgcalValidator.clone(
    LayerClustersInputMask = cms.VInputTag("hltTiclTrackstersCLUE3DHigh", "hltTiclSimTracksters:fromCPs", "hltTiclSimTracksters"),
    label_tst = [label for label in _hltTiclIterLabels] + ['hltTiclSimTracksters:fromCPs', 'hltTiclSimTracksters'],
    allTracksterTracksterAssociatorsLabels = ['hltAllTrackstersToSimTrackstersAssociationsByLCs:'+associator for associator in hltAssociatorInstances],
    allTracksterTracksterByHitsAssociatorsLabels = ['hltAllTrackstersToSimTrackstersAssociationsByHits:'+associator for associator in hltAssociatorInstances],
    associator = cms.untracked.InputTag("hltHGCalLayerClusterCaloParticleAssociation"),
    associatorSim = cms.untracked.InputTag("hltHGCalLayerClusterSimClusterAssociation"),
    dirName = 'HLT/HGCAL/HGCalValidator/',
    hits = 'hltHGCalRecHitMapProducer:RefProdVectorHGCRecHitCollection',
    hitMap = 'hltHGCalRecHitMapProducer:hgcalRecHitMap',
    simTrackstersMap = 'hltTiclSimTracksters',
    label_layerClustersPlots = 'hltHgcalMergeLayerClusters',
    label_lcl = 'hltMergeLayerClusters',
    label_simTS = 'hltTiclSimTracksters',
    label_simTSFromCP = 'hltTiclSimTracksters:fromCPs',
    recoTracks = 'hltGeneralTracks',
    simClustersToCaloParticlesMap = 'SimClusterToCaloParticleAssociation:simClusterToCaloParticleMap',
    simTiclCandidates = 'hltTiclSimTracksters',
    ticlCandidates = 'hltTiclCandidate',
    ticlTrackstersMerge = 'hltTiclTrackstersMerge',
    mergeRecoToSimAssociator = 'hltAllTrackstersToSimTrackstersAssociationsByLCs:hltTiclTrackstersMergeTohltTiclSimTrackstersfromCPs',
    mergeSimToRecoAssociator = 'hltAllTrackstersToSimTrackstersAssociationsByLCs:hltTiclSimTrackstersfromCPsTohltTiclTrackstersMerge',
)

from Configuration.ProcessModifiers.ticl_v5_cff import ticl_v5

lcInputMask_v5  = ["hltTiclTrackstersCLUE3DHigh"]
lcInputMask_v5.extend([cms.InputTag("hltTiclSimTracksters", "fromCPs"), cms.InputTag("hltTiclSimTracksters")])

ticl_v5.toModify(hltHgcalValidator,
                 LayerClustersInputMask = cms.VInputTag(lcInputMask_v5),
                 ticlTrackstersMerge = cms.InputTag("hltTiclCandidate"),
                 isticlv5 = cms.untracked.bool(True),
                 mergeSimToRecoAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs:hltTiclSimTrackstersfromCPsTohltTiclCandidate"),
                 mergeRecoToSimAssociator = cms.InputTag("hltAllTrackstersToSimTrackstersAssociationsByLCs:hltTiclCandidateTohltTiclSimTrackstersfromCPs"),
                 )

