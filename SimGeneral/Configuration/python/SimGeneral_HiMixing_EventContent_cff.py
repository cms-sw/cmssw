import FWCore.ParameterSet.Config as cms

HiMixRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
#        'keep *_mix_MergedTrackTruth_*',
#        'drop CrossingFramePlaybackInfoExtended_mix_*_*',
#        'keep *_mix_*_SIM',
    )
)

HiMixRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
#        'keep *_mix_MergedTrackTruth_*',
#        'drop CrossingFramePlaybackInfoExtended_mix_*_*',
#        'keep *_mix_*_SIM',
    )
)

HiMixAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
    )
)

