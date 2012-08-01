import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonPostprocessing_cfi import *
photonPostprocessing.batch = cms.bool(False)
photonPostprocessing.standalone = cms.bool(False)
photonPostprocessing.isRunCentrally = cms.bool(True)

from Validation.RecoEgamma.conversionPostprocessing_cfi import *
conversionPostprocessing.batch = cms.bool(False)
conversionPostprocessing.standalone = cms.bool(False)
conversionPostprocessing.fastSim = cms.bool(True)

photonPostProcessor = cms.Sequence(photonPostprocessing)
