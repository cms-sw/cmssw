import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.PostProcessorBarrel_cfi import postProcessorBarrellayerclusters, postProcessorBarrelTracksters

barrelValidatorPostProcessor = cms.Sequence()
_barrelValidatorPostProcessor = barrelValidatorPostProcessor.copy()
_barrelValidatorPostProcessor += cms.Sequence(postProcessorBarrellayerclusters
                                              +postProcessorBarrelTracksters
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(barrelValidatorPostProcessor, _barrelValidatorPostProcessor)
