import FWCore.ParameterSet.Config as cms

from FWCore.Modules.printContent_cfi import * 

from Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi import * 
from Validation.RecoEgamma.ElectronIsolation_cfi import *
 
from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import egmElectronIsolationCITK as _egmElectronIsolationCITK
from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *

miniAODElectronIsolation = _egmElectronIsolationCITK.clone( 
    srcToIsolate = "slimmedElectrons",
    srcForIsolationCone = "packedPFCandidates"
)

electronValidationTaskMiniAOD = cms.Task(egmElectronIsolationCITK, miniAODElectronIsolation, ElectronIsolation)
electronValidationSequenceMiniAOD = cms.Sequence(electronMcSignalValidatorMiniAOD, electronValidationTaskMiniAOD)

