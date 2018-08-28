import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitStudy_cfi import *

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalSimHitStudy,
         detectorNames = cms.vstring("HGCalEESensitive",
                                     "HGCalHESiliconSensitive",
                                     "HGCalHEScintillatorSensitive"),
                        caloHitSources = cms.vstring('HGCHitsEE',
                                                     'HGCHitsHEfront',
                                                     'HGCHitsHEback'
                                                     )
                        )
