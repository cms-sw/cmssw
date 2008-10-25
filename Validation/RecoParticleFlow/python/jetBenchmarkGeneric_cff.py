import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfJetBenchmarkGeneric_cfi import pfJetBenchmarkGeneric
from Validation.RecoParticleFlow.caloJetBenchmarkGeneric_cfi import caloJetBenchmarkGeneric


jetBenchmarkGeneric = cms.Sequence( 
    pfJetBenchmarkGeneric +
    caloJetBenchmarkGeneric
    )
