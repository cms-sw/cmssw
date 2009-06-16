import FWCore.ParameterSet.Config as cms

process = cms.Process("PRODMIX")

process.load("SimGeneral.DataMixingModule.mixOne_data_on_sim_cfi")

#-------------------------------------------------------
#
#modify Digi sequences to regenerate Trigger primitives for calorimeter
#
process.load('Configuration/StandardSequences/DigiDM_cff')
#
#

#process.pMix = cms.Path(process.mix+process.mixData)
process.DataMix = cms.Path(process.mixData+process.PostDM_Digi)
