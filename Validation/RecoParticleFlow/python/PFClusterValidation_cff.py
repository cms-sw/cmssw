import FWCore.ParameterSet.Config as cms
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation
from Validation.RecoParticleFlow.pfCaloGPUComparisonTask_cfi import pfClusterHBHEOnlyAlpakaComparison, pfClusterHBHEAlpakaComparison

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterAlpakaComparisonSequence = cms.Sequence( pfClusterHBHEAlpakaComparison )

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)

pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )

pfClusterHBHEOnlyAlpakaComparisonSequence = cms.Sequence( pfClusterHBHEOnlyAlpakaComparison )
