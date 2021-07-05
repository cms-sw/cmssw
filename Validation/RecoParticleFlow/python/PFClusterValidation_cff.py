import FWCore.ParameterSet.Config as cms
from Validation.RecoParticleFlow.pfClusterValidation_cfi import pfClusterValidation

pfClusterValidationSequence = cms.Sequence( pfClusterValidation )

pfClusterCaloOnlyValidation = pfClusterValidation.clone(
    pflowClusterHCAL = 'particleFlowClusterHCALOnly'
)

pfClusterCaloOnlyValidationSequence = cms.Sequence( pfClusterCaloOnlyValidation )
