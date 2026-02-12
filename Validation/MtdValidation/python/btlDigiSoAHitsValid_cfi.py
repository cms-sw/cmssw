import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.btlDigiSoAHitsDefaultValid_cfi import btlDigiSoAHitsDefaultValid as _btlDigiSoAHitsDefaultValid
btlDigiSoAHitsValid = _btlDigiSoAHitsDefaultValid.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(btlDigiSoAHitsValid, inputTag = "mixData:FTLBarrelSoA")