import FWCore.ParameterSet.Config as cms

#
#
# event content for HEEP Express Stream Skim
#
#
from Configuration.EventContent.EventContent_cff import *
AODSIMHEEPExpressEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
hEEPExpressEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring()
    )
)
AODSIMHEEPExpressEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
hEEPExpressEventSelection.SelectEvents.SelectEvents.append('hEEPSignalHighEt')
hEEPExpressEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtHigh')
hEEPExpressEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtMedBarrel')
hEEPExpressEventSelection.SelectEvents.SelectEvents.append('hEEPSignalMedEtMedEndcap')

