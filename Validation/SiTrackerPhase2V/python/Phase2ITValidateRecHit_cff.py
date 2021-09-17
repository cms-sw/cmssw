import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2ITValidateRecHit_cfi import Phase2ITValidateRecHit
rechitValidIT = Phase2ITValidateRecHit.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(rechitValidIT,
    pixelSimLinkSrc = "mixData:PixelDigiSimLink",
)
