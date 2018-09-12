import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudy_cfi import *

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalDigiStudy,
                        detectorNames = cms.vstring("HGCalEESensitive",
                                                    "HGCalHESiliconSensitive",
                                                    "HGCalHEScintillatorSensitive")
                        )
