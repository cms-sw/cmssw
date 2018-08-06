import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalBHAnalysis_cfi import *

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalBHAnalysis,
         GeometryType  = cms.untracked.int32(1),
         HitCollection = cms.untracked.string("HGCHitsHEback"),
)
