import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyMuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyMuonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyMuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyMuonEventSelection.SelectEvents.SelectEvents.append('hadSUSYdiMuon')
susyMuonEventSelection.SelectEvents.SelectEvents.append('hadSUSYTopMuon')
susyMuonEventSelection.SelectEvents.SelectEvents.append('hadSUSYQCDControlMET')
susyMuonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_1Muon_0Elec_1Jets_MET')
susyMuonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_1Muon_1Elec_2Jets_MET')
susyMuonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_2Muon_0Elec_2Jets_MET')
susyMuonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_1Muon_0Elec_2Jets_MET')

