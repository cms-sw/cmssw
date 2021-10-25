import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2TrackerValidateDigi_cff import *
from Validation.SiTrackerPhase2V.Phase2ITValidateRecHit_cff import *
from Validation.SiTrackerPhase2V.Phase2ITValidateTrackingRecHit_cff import *
from Validation.SiTrackerPhase2V.Phase2ITValidateCluster_cff import *
from Validation.SiTrackerPhase2V.Phase2OTValidateCluster_cff import *
from Validation.SiTrackerPhase2V.Phase2OTValidateTrackingRecHit_cff import *

trackerphase2ValidationSource = cms.Sequence(pixDigiValid  
                                             + otDigiValid 
                                             + rechitValidIT
                                             + trackingRechitValidIT
                                             + clusterValidIT
                                             + clusterValidOT
                                             + trackingRechitValidOT
)

from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toReplaceWith(trackerphase2ValidationSource, trackerphase2ValidationSource.copyAndExclude([trackingRechitValidOT]))
