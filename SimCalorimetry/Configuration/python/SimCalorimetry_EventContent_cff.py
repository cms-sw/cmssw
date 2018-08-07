import FWCore.ParameterSet.Config as cms

# This object modifies the event content for different scenarios

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
        'keep *_mixData_HcalSamples_*',
        'keep *_mix_HcalHits_*',
        'keep *_mixData_HcalHits_*',
        'keep *_DMHcalTriggerPrimitiveDigis_*_*',
    )
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
SimCalorimetryPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep EBDigiCollection_simEcalDigis_*_*',
        'keep EEDigiCollection_simEcalDigis_*_*',
        'keep ESDigiCollection_simEcalUnsuppressedDigis_*_*',
        'keep *_simHcalDigis_*_*',
    )
)

#
# Add extra event content if running in Run 2
#
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*') )
run2_common.toModify( SimCalorimetryRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simHcalUnsuppressedDigis_*_*') )

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_DMHcalDigis_*_*') )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_mix_EETimeDigi_*') )
phase2_timing.toModify(SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_mix_EBTimeDigi_*') )

phase2_timing.toModify(SimCalorimetryRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_mix_EETimeDigi_*') )
phase2_timing.toModify(SimCalorimetryRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_mix_EBTimeDigi_*') )

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify( SimCalorimetryFEVTDEBUG.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simEcalUnsuppressedDigis_*_*') )
phase2_common.toModify( SimCalorimetryRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simEcalUnsuppressedDigis_*_*') )
phase2_common.toModify( SimCalorimetryPREMIX.outputCommands, func=lambda outputCommands: outputCommands.append('drop ESDigiCollection_simEcalUnsuppressedDigis_*_*') )

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( SimCalorimetryPREMIX.outputCommands, func=lambda outputCommands: outputCommands.append('drop EEDigiCollection_simEcalDigis_*_*') )
