import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.PFMETFilter_cfi import metFilter

Filter = cms.Sequence(
    metFilter
    )

