import FWCore.ParameterSet.Config as cms

# the cluster shape limits are read from SiPixelClusterShapeLimitsRcd (Global Tag)
from RecoTracker.PixelLowPtUtilities.clusterShapeHitFilterESProducer_cfi import clusterShapeHitFilterESProducer
ClusterShapeHitFilterESProducer = clusterShapeHitFilterESProducer.clone(ComponentName = 'ClusterShapeHitFilter',
                                                                        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
                                                                        isPhase2 = False)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(ClusterShapeHitFilterESProducer,
    isPhase2 = True,
)
