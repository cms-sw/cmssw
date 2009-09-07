import FWCore.ParameterSet.Config as cms

HiMixRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_hiSignal_*_*',
        'keep *_hiSignalG4SimHits_*_*'
    )
)

HiMixRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_hiSignal_*_*'
    )
)

HiMixAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)



