import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.btlDigiHitsDefault_cfi import btlDigiHitsDefault as _btlDigiHitsDefault
btlDigiHits = _btlDigiHitsDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(btlDigiHits, inputTag = "mixData:FTLBarrel")
