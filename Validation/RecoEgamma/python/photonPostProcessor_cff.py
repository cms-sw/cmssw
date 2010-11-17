import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonPostprocessing_cfi import *
photonPostprocessing.batch = cms.bool(False)
photonPostprocessing.standalone = cms.bool(False)

from Validation.RecoEgamma.conversionPostprocessing_cfi import *
conversionPostprocessing.batch = cms.bool(False)
conversionPostprocessing.standalone = cms.bool(False)

photonPostProcessor = cms.Sequence(photonPostprocessing*conversionPostprocessing)
