import FWCore.ParameterSet.Config as cms

# This object modifies the event content for different scenarios
from Configuration.StandardSequences.Eras import eras

SimCalorimetryFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_simEcalDigis_*_*', 
        'keep *_simEcalPreshowerDigis_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*', 
        'keep *_simEcalEBTriggerPrimitiveDigis_*_*', 
        'keep *_simHcalDigis_*_*', 
        'keep ZDCDataFramesSorted_simHcalUnsuppressedDigis_*_*',
        'drop ZDCDataFramesSorted_mix_simHcalUnsuppressedDigis*_*',
        'keep *_simHcalTriggerPrimitiveDigis_*_*',
        'keep *_mix_HcalSamples_*',
        'keep *_mix_HcalHits_*')
)
SimCalorimetryRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep EBSrFlagsSorted_simEcalDigis_*_*', 
        'keep EESrFlagsSorted_simEcalDigis_*_*')
)
SimCalorimetryRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
SimCalorimetryAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

#
# Add extra event content if running in Run 2
#
eras.run2_common.toModify( SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*') )
eras.run2_common.toModify( SimCalorimetryRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*') )
