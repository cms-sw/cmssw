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

hltLayerClusterTesterECAL = cms.EDProducer("CaloClusterTester",
    PFCand = cms.InputTag("hltParticleFlowTmp"),
    Rechit = cms.InputTag("hltParticleFlowRecHitECALUnseeded"),
    RecoCluster = cms.InputTag("hltBarrelLayerClustersEB"),
    SimCluster = cms.InputTag("mix","MergedCaloTruth"),
    CaloParticle = cms.InputTag("mix","MergedCaloTruth"),
    ClusterSimClusterAssociator = cms.InputTag("hltBarrelLayerClusterSimClusterAssociationProducer"),
    ClusterCaloParticleAssociator = cms.InputTag("hltBarrelLayerClusterCaloParticleAssociationProducer"),
    outFolder = cms.string('HLT/TiclBarrel'),
    assocScoreThresholds = cms.vdouble(1., 0.5, 0.1),
    doMatchByScore = cms.bool(True),
    enFracCut = cms.double(0.),
    ptCut = cms.double(0.),
    etaCut = cms.double(1.48)
)

hltLayerClusterTesterECALWithCut = hltLayerClusterTesterECAL.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.1)
)

# SimToReco match based on shared energy fraction
hltLayerClusterTesterECALShEnF = hltLayerClusterTesterECAL.clone(
    doMatchByScore = cms.bool(False)
)

hltLayerClusterTesterECALShEnFWithCut = hltLayerClusterTesterECALShEnF.clone(
    enFracCut =  cms.double(0.01),
    ptCut = cms.double(0.1)
)


hltHgcalValSeq = cms.Sequence(
    hltHgcalValidator)

hltHgcalAndBarrelValSeq = cms.Sequence(
    hltHgcalValidator
    +hltLayerClusterTesterECALWithCut
    +hltLayerClusterTesterECALShEnFWithCut
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(hltHgcalValSeq, hltHgcalAndBarrelValSeq)
