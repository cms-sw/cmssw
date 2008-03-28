# The following comments couldn't be translated into the new config version:

#save trigger primitive digi and ecal SrFlags

#save trigger primitive digi and ecal SrFlags
#save digis, but not ECAL Unsuppressed ones

import FWCore.ParameterSet.Config as cms

#Full Event content 
SimCalorimetryFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep EBSrFlagsSorted_*_*_*', 'keep EESrFlagsSorted_*_*_*')
)
#Full Event content with DIGI
SimCalorimetryFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalDigis_*_*', 'keep *_ecalPreshowerDigis_*_*', 'keep *_ecalTriggerPrimitiveDigis_*_*', 'keep *_hcalDigis_*_*', 'keep *_hcalTriggerPrimitiveDigis_*_*')
)
#RECO content
SimCalorimetryRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#AOD content
SimCalorimetryAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

