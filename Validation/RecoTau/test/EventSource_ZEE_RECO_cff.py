import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
   '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/FC71916C-756B-DE11-8631-000423D94700.root',
   '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/B672A1C5-746B-DE11-93A8-000423D944F0.root',
   '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/5A0F871C-756B-DE11-BFE1-001D09F2545B.root',
   '/store/relval/CMSSW_3_1_1/RelValZEE/GEN-SIM-RECO/MC_31X_V2-v1/0002/44507D17-D66B-DE11-A165-000423D94524.root' ] );

secFiles.extend( [
   ] )


