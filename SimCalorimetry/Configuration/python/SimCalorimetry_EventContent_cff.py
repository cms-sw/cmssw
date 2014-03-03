import FWCore.ParameterSet.Config as cms

SimCalorimetryFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_simEcalDigis_*_*', 
        'keep *_simEcalPreshowerDigis_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*', 
        'keep *_simHcalDigis_*_*', 
        'keep ZDCDataFramesSorted_simHcalUnsuppressedDigis_*_*',
        'drop ZDCDataFramesSorted_mix_simHcalUnsuppressedDigis*_*',
        'keep *_simHcalTriggerPrimitiveDigis_*_*')
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

