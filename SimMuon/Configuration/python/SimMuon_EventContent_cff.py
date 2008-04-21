# The following comments couldn't be translated into the new config version:

#save digis sim link and trigger infos

#save digis

import FWCore.ParameterSet.Config as cms

#Full Event content 
SimMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep StripDigiSimLinkedmDetSetVector_muonCSCDigis_*_*', 
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_*_*')
)
#Full Event content with DIGI
SimMuonFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*')
)
#RECO content
SimMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#AOD content
SimMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

