import FWCore.ParameterSet.Config as cms

HiMixRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep CrossingFramePlaybackInfoNew_mix_*_*',
        'keep *_heavyIon_*_*',
    )
)

HiMixRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep CrossingFramePlaybackInfoNew_mix_*_*',
        'keep *_heavyIon_*_*',
    )
)

HiMixAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep CrossingFramePlaybackInfoNew_mix_*_*',
        'keep *_heavyIon_*_*'
    )
)

