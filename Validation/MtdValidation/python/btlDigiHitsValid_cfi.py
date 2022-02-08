import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.btlDigiHitsDefaultValid_cfi import btlDigiHitsDefaultValid as _btlDigiHitsDefaultValid
btlDigiHitsValid = _btlDigiHitsDefaultValid.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(btlDigiHitsValid, inputTag = "mixData:FTLBarrel")
