import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.etlDigiHitsDefaultValid_cfi import etlDigiHitsDefaultValid as _etlDigiHitsDefaultValid
etlDigiHitsValid = _etlDigiHitsDefaultValid.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(etlDigiHitsValid, inputTag = "mixData:FTLEndcap")
