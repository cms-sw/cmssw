import FWCore.ParameterSet.Config as cms

from SimCalorimetry.TICLSimProducers.simTrackstersProducer_cfi import simTrackstersProducer as _simTrackstersProducer
from RecoTICL.ClusterFilters.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer
from Validation.RecoTrack.associators_cff import hltTrackAssociatorByHits, tpToHLTpixelTrackAssociation, tpToHLTgsfTrackAssociation
from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import simHitTPAssocProducer
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation
# HLT TICL geometry for the HLT validation (see _setHltTiclGeom below)
from HLTrigger.Configuration.HLT_75e33.eventsetup.hltTiclGeomESProducer_cfi import hltTiclGeomESProducer as _hltTiclGeomESProducer
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common as _phase2_common
# CA - PATTERN RECOGNITION

hltFilteredLayerClustersSimTracksters = _filteredLayerClustersProducer.clone(
    ticlGeom = cms.ESInputTag('', 'hlt'),
    LayerClusters = cms.InputTag("hltMergeLayerClusters"),
    LayerClustersInputMask = cms.InputTag("hltMergeLayerClusters","InitialLayerClustersMask"),
    clusterFilter = "ClusterFilterByAlgoAndSize",
    min_cluster_size = 0, # inclusive
    iteration_label = "hltTiclSimTracksters"
)

tpToHltGeneralTrackAssociation = tpToHLTpixelTrackAssociation.clone(
    label_tr = "hltGeneralTracks"
)

tpHltGsfTrackAssociation = tpToHLTgsfTrackAssociation.clone(
    label_tr = cms.InputTag("hltEgammaGsfTracksL1Seeded"),
)

hltTiclSimTracksters = _simTrackstersProducer.clone(
    ticlGeom = cms.ESInputTag('', 'hlt'),
    layerClusterCaloParticleAssociator = cms.InputTag("hltHGCalLayerClusterCaloParticleAssociation"),
    layerClusterSimClusterAssociator = cms.InputTag("hltHGCalLayerClusterSimClusterAssociation"),
    filtered_mask = cms.InputTag("hltFilteredLayerClustersSimTracksters","hltTiclSimTracksters"),
    layer_clusters = cms.InputTag("hltMergeLayerClusters"),
    time_layerclusters = cms.InputTag("hltMergeLayerClusters","timeLayerCluster"),
    simTrackToTPMap = cms.InputTag("simHitTPAssocProducer","simTrackToTP"),
    recoTracks = cms.InputTag("hltGeneralTracks"),
    gsfTracks  = cms.InputTag("hltEgammaGsfTracksL1Seeded"),
    simclusters = cms.InputTag("mix","MergedCaloTruth"),
    tpToTrack = cms.InputTag("tpToHltGeneralTrackAssociation"),
    tpToGsfTrack  = cms.InputTag("tpHltGsfTrackAssociation"),
    computeLocalTime = cms.bool(True)
)

from Validation.Configuration.hltHGCalSimValid_cff import *

# The HLT validation consumes the HLT TICL geometry (data labels "hlt", "hltWithBarrel").
# Decide where it comes from once the validation is loaded, so that a job holds one copy only:
#  - HLT menu in the same job: use the menu's own instance;
#  - offline reconstruction in the same job (and no HLT menu): the HLT and offline geometries
#    are the same, so point the HLT validation modules at the offline labels instead of
#    building a second copy;
#  - neither (validation-only job): attach the HLT instance.
# The producer is attached outside of any Task: an EventSetup module that belongs to a Task is
# enabled only if that Task is scheduled, which would switch off the HLT menu's own instance in
# jobs that load but do not schedule the HLT validation.
_hltToOfflineTiclGeomLabel = {'hlt': '', 'hltWithBarrel': 'withBarrel'}

def _setHltTiclGeom(process):
    if hasattr(process, 'hltTiclGeomESProducer'):
        return
    if hasattr(process, 'ticlGeomESProducer'):
        for module in list(process.producers_().values()) + list(process.analyzers_().values()):
            for par in ('ticlGeom', 'ticlGeomWithBarrel'):
                tag = getattr(module, par, None)
                if isinstance(tag, cms.ESInputTag) and tag.getDataLabel() in _hltToOfflineTiclGeomLabel:
                    setattr(module, par, cms.ESInputTag('', _hltToOfflineTiclGeomLabel[tag.getDataLabel()]))
        return
    process.hltTiclGeomESProducer = _hltTiclGeomESProducer.clone()

setHltTiclGeomForHLTValidation = _phase2_common.makeProcessModifier(_setHltTiclGeom)

hltTiclSimTrackstersTask = cms.Task(hltTrackAssociatorByHits,
                                    tpToHltGeneralTrackAssociation,
                                    tpHltGsfTrackAssociation,
                                    simHitTPAssocProducer,
                                    hltHgcalAssociatorsTask,
                                    hltFilteredLayerClustersSimTracksters,
                                    hltTiclSimTracksters)

hltTiclSimTrackstersSeq = cms.Sequence(
    hltTiclSimTrackstersTask
)
