import FWCore.ParameterSet.Config as cms

from Validation.SiTrackerPhase2V.Phase2OTValidateRecHit_cfi import * 

rechitValidOT = Phase2OTValidateRecHit.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(rechitValidOT,
    phase2TrackerSimLinkSrc = "mixData:Phase2OTDigiSimLink",
)
