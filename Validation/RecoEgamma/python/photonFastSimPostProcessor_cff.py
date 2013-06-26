import FWCore.ParameterSet.Config as cms

#from Validation.RecoEgamma.photonPostprocessing_cfi import *
import Validation.RecoEgamma.photonPostprocessing_cfi 
fastSimPhotonPostProcessing=Validation.RecoEgamma.photonPostprocessing_cfi.photonPostprocessing.clone()

fastSimPhotonPostProcessing.batch = cms.bool(False)
fastSimPhotonPostProcessing.standalone = cms.bool(False)
fastSimPhotonPostProcessing.isRunCentrally = cms.bool(True)
fastSimPhotonPostProcessing.fastSim = cms.bool(True)


#from Validation.RecoEgamma.conversionPostprocessing_cfi import *
#conversionPostprocessing.batch = cms.bool(False)
#conversionPostprocessing.standalone = cms.bool(False)
#conversionPostprocessing.fastSim = cms.bool(True)

fastSimPhotonPostProcessor = cms.Sequence(fastSimPhotonPostProcessing)
