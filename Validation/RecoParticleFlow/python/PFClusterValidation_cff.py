import FWCore.ParameterSet.Config as cms
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation
from Validation.RecoParticleFlow.PFCaloGPUComparisonTask_cfi import pfHBHEGPUComparisonTask

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)

pfHBHEGPUComparisonTaskCaloOnly = pfHBHEGPUComparisonTask.clone(
    pfClusterToken_ref = 'particleFlowClusterHBHEOnly@cpu',
    pfClusterToken_target = 'particleFlowClusterHBHEOnly@cuda'
)

pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )
pfClusterCaloOnlyValidationSequenceGPU = cms.Sequence( pfClusterCaloOnlyValidation + pfHBHEGPUComparisonTaskCaloOnly)

from Configuration.ProcessModifiers.gpuValidationPF_cff import gpuValidationPF
gpuValidationPF.toReplaceWith(pfClusterCaloOnlyValidationSequence, pfClusterCaloOnlyValidationSequenceGPU)
