import FWCore.ParameterSet.Config as cms

from Validation.RecoEgamma.photonPostprocessing_cfi import *
import Validation.RecoEgamma.photonPostprocessing_cfi 


photonPostprocessing.batch = cms.bool(False)
photonPostprocessing.standalone = cms.bool(False)
photonPostprocessing.isRunCentrally = cms.bool(True)
#
pfPhotonPostprocessing = Validation.RecoEgamma.photonPostprocessing_cfi.photonPostprocessing.clone()
pfPhotonPostprocessing.ComponentName = cms.string('pfPhotonPostprocessing')
pfPhotonPostprocessing.analyzerName = cms.string('pfPhotonValidator')
pfPhotonPostprocessing.batch = cms.bool(False)
pfPhotonPostprocessing.standalone = cms.bool(False)
pfPhotonPostprocessing.isRunCentrally = cms.bool(True)
#
oldpfPhotonPostprocessing = Validation.RecoEgamma.photonPostprocessing_cfi.photonPostprocessing.clone()
oldpfPhotonPostprocessing.ComponentName = cms.string('oldpfPhotonPostprocessing')
oldpfPhotonPostprocessing.analyzerName = cms.string('oldpfPhotonValidator')
oldpfPhotonPostprocessing.batch = cms.bool(False)
oldpfPhotonPostprocessing.standalone = cms.bool(False)
oldpfPhotonPostprocessing.isRunCentrally = cms.bool(True)
#
from Validation.RecoEgamma.conversionPostprocessing_cfi import *
conversionPostprocessing.batch = cms.bool(False)
conversionPostprocessing.standalone = cms.bool(False)

photonPostProcessor = cms.Sequence(photonPostprocessing*pfPhotonPostprocessing*oldpfPhotonPostprocessing*conversionPostprocessing)
#photonPostProcessor = cms.Sequence(photonPostprocessing*conversionPostprocessing)

