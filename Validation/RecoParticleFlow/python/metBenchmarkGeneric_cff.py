import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfMETBenchmarkGeneric_cfi import pfMETBenchmarkGeneric
from Validation.RecoParticleFlow.caloMETBenchmarkGeneric_cfi import caloMETBenchmarkGeneric

metBenchmarkGeneric = cms.Sequence( 
    pfMETBenchmarkGeneric
    +
    caloMETBenchmarkGeneric
    )
# foo bar baz
# 5NUdEElcVdvcg
# dCx6oK2JvzIfJ
