import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
AODSIMSusyElectronPhotonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
susyElectronPhotonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMSusyElectronPhotonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_1Elec_1Jets_MET')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_2Elec_2Jets_MET')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_1Muon_1Elec_2Jets_MET')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('lepSUSY_0Muon_1Elec_2Jets_MET')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hadSUSYdiElec')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hadSUSYTopElec')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('SUSYHighPtPhoton')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('SUSYControlHighPtPhoton')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hEEPSignalHighEt')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtHigh')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtMedBarrel')
susyElectronPhotonEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtMedEndcap')

