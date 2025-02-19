import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.Filter_cfi import metFilter

Filter = cms.Sequence(
    metFilter
    )

