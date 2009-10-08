import FWCore.ParameterSet.Config as cms


from Validation.RecoParticleFlow.pfCandidateBenchmark_cfi import pfCandidateBenchmark


# could create one benchmark / particle type

pfCandidateBenchmarkSequence = cms.Sequence(
    pfCandidateBenchmark
    )
