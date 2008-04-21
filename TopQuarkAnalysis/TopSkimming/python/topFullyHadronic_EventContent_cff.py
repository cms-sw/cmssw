import FWCore.ParameterSet.Config as cms

topFullyHadronicEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
topFullyHadronicJetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicJetsPath')
    )
)
topFullyHadronicBJetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicBJetsPath')
    )
)
topFullyHadronicEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicJetsPath', 
            'topFullyHadronicBJetsPath')
    )
)

