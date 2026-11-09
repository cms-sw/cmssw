import FWCore.ParameterSet.Config as cms

# Alpaka ESProducer -- framework auto-selects backend (CPU serial, CUDA, ROCm).
stackedModuleGeometryESProducer = cms.ESProducer('StackedModuleGeometryESProducer@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string('')  # Empty string = use default backend
    )
)
