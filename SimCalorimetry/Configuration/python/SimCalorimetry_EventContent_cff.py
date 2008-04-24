# The following comments couldn't be translated into the new config version:

#save trigger primitive digi and ecal SrFlags

#save trigger primitive digi and ecal SrFlags
#save digis, but not ECAL Unsuppressed ones

import FWCore.ParameterSet.Config as cms

#Full Event content 
SimCalorimetryFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep EBSrFlagsSorted_simEcalDigis_*_*', 
        'keep EESrFlagsSorted_simEcalDigis_*_*')
)
#Full Event content with DIGI
SimCalorimetryFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_simEcalDigis_*_*', 
        'keep *_simEcalPreshowerDigis_*_*', 
        'keep *_simEcalTriggerPrimitiveDigis_*_*', 
        'keep *_simHcalDigis_*_*', 
        'keep *_simHcalTriggerPrimitiveDigis_*_*')
)
#RECO content
SimCalorimetryRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#AOD content
SimCalorimetryAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

