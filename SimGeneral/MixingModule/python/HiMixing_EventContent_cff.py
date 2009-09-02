import FWCore.ParameterSet.Config as cms

HiMixSIM = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hiSignal_*_*',
                                           'keep *_hiSignalG4SimHits_*_*'
                                           )
    )
