import FWCore.ParameterSet.Config as cms
from Validation.MtdValidation.etlDigiHitsDefault_cfi import etlDigiHitsDefault as _etlDigiHitsDefault
etlDigiHits = _etlDigiHitsDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(etlDigiHits, inputTag = "mixData:FTLEndcap")
