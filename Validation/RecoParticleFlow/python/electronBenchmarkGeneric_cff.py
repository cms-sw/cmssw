import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.electronBenchmarkGeneric_cfi import electronBenchmarkGeneric

# add here specific things needed for the electron benchmark if needed

electronBenchmarkGeneric = cms.Sequence( 
    electronBenchmarkGeneric
    )
