import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2ITValidateTrackingRecHit_cfi import Phase2ITValidateTrackingRecHit
trackingRechitValidIT = Phase2ITValidateTrackingRecHit.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingRechitValidIT,
    pixelSimLinkSrc = "mixData:PixelDigiSimLink",
)
