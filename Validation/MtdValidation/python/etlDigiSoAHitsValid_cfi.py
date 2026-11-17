import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.etlDigiSoAHitsDefaultValid_cfi import etlDigiSoAHitsDefaultValid as _etlDigiSoAHitsDefaultValid
etlDigiSoAHitsValid = _etlDigiSoAHitsDefaultValid.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(etlDigiSoAHitsValid, inputTag = "mixData:FTLEndcapSoA")
