import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalShowerSeparationDefault_cfi import hgcalShowerSeparationDefault as _hgcalShowerSeparationDefault
hgcalShowerSeparation = _hgcalShowerSeparationDefault.clone()

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hgcalShowerSeparation, caloParticles = "mixData:MergedCaloTruth")
