import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyJetMETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyJetMETEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyJetMETEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyJetMETEventSelection.SelectEvents.SelectEvents.append('hadSUSYQCDControlMET')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('hadSUSYQCD')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('hadSUSYSMBackgr')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_1Elec_2Jets_MET')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('lepSUSY_1Muon_0Elec_2Jets_MET')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_1Elec_1Jets_MET')
susyJetMETEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_2Elec_2Jets_MET')

