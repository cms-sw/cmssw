import FWCore.ParameterSet.Config as cms

# this is a minimum configuration of the Mixing module,
# to run it in the zero-pileup mode
#
mix = cms.EDFilter("MixingModule",
    bunchspace = cms.int32(25),
    playback = cms.untracked.bool(False),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    Label = cms.string('')
)


