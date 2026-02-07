import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2ITValidateCluster_cff import *
from Validation.SiTrackerPhase2V.Phase2OTValidateCluster_cff import *
from Validation.SiTrackerPhase2V.Phase2ITValidateRecHit_cff import *
from Validation.SiTrackerPhase2V.Phase2OTValidateRecHit_cff import *
from Validation.SiTrackerPhase2V.Phase2ITValidateTrackingRecHit_cff import *
from Validation.SiTrackerPhase2V.Phase2OTValidateTrackingRecHit_cff import *

hltClusterValidIT = clusterValidIT.clone(
    ClusterSource = "hltSiPixelClusters",
    TopFolderName = 'HLT/TrackerPhase2ITClusterV'
)

hltClusterValidOT = clusterValidOT.clone(
    ClusterSource = "hltSiPhase2Clusters",
    TopFolderName = 'HLT/TrackerPhase2OTClusterV'
)

hltRechitValidIT = rechitValidIT.clone(
    rechitsSrc = "hltSiPixelRecHits",
    TopFolderName = 'HLT/TrackerPhase2ITRecHitV',
)

hltRechitValidOT = rechitValidOT.clone(
    rechitsSrc = "hltSiPhase2RecHits",
    TopFolderName = 'HLT/TrackerPhase2OTRecHitV',
)

hltTrackingRechitValidIT = trackingRechitValidIT.clone(
    tracksSrc = "hltGeneralTracks",
    TopFolderName = 'HLT/TrackerPhase2ITTrackingRecHitV'
)

hltTrackingRechitValidOT = trackingRechitValidOT.clone(
    tracksSrc = "hltGeneralTracks",
    TopFolderName = 'HLT/TrackerPhase2OTTrackingRecHitV'
)

hltTrackerphase2ValidationSource = cms.Sequence(hltClusterValidIT + 
                                                hltClusterValidOT +
                                                hltRechitValidIT  +
                                                hltTrackingRechitValidIT +
                                                hltTrackingRechitValidOT)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
(~hltPhase2LegacyTracking).toModify(
    hltTrackerphase2ValidationSource,
    lambda s: s.__iadd__(hltRechitValidOT)
)
