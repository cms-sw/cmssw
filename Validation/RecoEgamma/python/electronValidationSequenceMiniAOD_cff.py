import FWCore.ParameterSet.Config as cms

from FWCore.Modules.printContent_cfi import * 

from Validation.RecoEgamma.ElectronMcSignalValidatorMiniAOD_cfi import * 
from Validation.RecoEgamma.ElectronIsolation_cfi import *
 
from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import egmElectronIsolationCITK as _egmElectronIsolationCITK
from RecoParticleFlow.PFProducer.particleFlowEGamma_cff import *

miniAODElectronIsolation = _egmElectronIsolationCITK.clone() 
miniAODElectronIsolation.srcToIsolate = cms.InputTag("slimmedElectrons")
miniAODElectronIsolation.srcForIsolationCone = cms.InputTag("packedPFCandidates")

electronValidationSequenceMiniAOD = cms.Sequence( egmElectronIsolationCITK + miniAODElectronIsolation + ElectronIsolation + electronMcSignalValidatorMiniAOD )

