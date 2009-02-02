import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfMetBenchmarkGeneric_cfi import pfMetBenchmarkGeneric
#from Validation.RecoParticleFlow.caloMetBenchmarkGeneric_cfi import caloMetBenchmarkGeneric

# add here specific things needed for the MET benchmark if needed
# for example, the computation of the GenMET. 

metBenchmarkGeneric = cms.Sequence( 
    pfMetBenchmarkGeneric
#    +
#    caloMetBenchmarkGeneric
    )
