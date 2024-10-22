import FWCore.ParameterSet.Config as cms

from Validation.SiTrackerPhase2V.Phase2OTValidateTrackingRecHit_cfi import * 

trackingRechitValidOT = Phase2OTValidateTrackingRecHit.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingRechitValidOT,
    phase2TrackerSimLinkSrc = "mixData:Phase2OTDigiSimLink",
)
